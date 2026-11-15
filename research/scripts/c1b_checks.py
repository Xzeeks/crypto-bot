#!/usr/bin/env python3
"""c1b_checks.py — follow-ups on the c1 first read.   python research/scripts/c1b_checks.py research/data/eth_above.duckdb

 1. Candle convention: does the 11:59 candle (closes AT 12:00) fit the recorded outcomes better than the 12:00 candle?
 2. Final-3h mid-price Yes overpricing, aggregated by event-day (the real n), with a sign test.
 3. Post-print flow by minute after the candle close: $ and the price the known winner trades at.
"""
import sys, math
import duckdb, numpy as np, pandas as pd
pd.set_option("display.width", 200)
con = duckdb.connect(sys.argv[1], read_only=True)

print("=== 1. candle convention ===")
for name, off in [("opens 12:00 (close 12:01)", 0), ("opens 11:59 (close 12:00)", -60), ("opens 12:01", 60)]:
    tot, bad = con.execute(f"""
        SELECT COUNT(*), SUM(CASE WHEN (k.close > m.strike) <> (m.outcome = 1) THEN 1 ELSE 0 END)
        FROM markets m JOIN klines k ON k.open_ts = m.resolve_ts + ({off})
        WHERE m.outcome IS NOT NULL""").fetchone()
    print(f"{name:28s} {tot} strikes, {bad} mismatches")
print(con.execute("""
    SELECT m.event_date, m.strike, m.outcome, k0.close AS close_1159, k1.close AS close_1200, k2.close AS close_1201
    FROM markets m JOIN klines k0 ON k0.open_ts = m.resolve_ts - 60
                   JOIN klines k1 ON k1.open_ts = m.resolve_ts
                   JOIN klines k2 ON k2.open_ts = m.resolve_ts + 60
    WHERE m.outcome IS NOT NULL AND ABS(k1.close - m.strike) < 5""").df().to_string(index=False))

print("\n=== 2. final-3h mid-price Yes (0.35-0.80), by event-day ===")
d = con.execute("""
    SELECT m.event_date, m.strike,
           CASE WHEN f.token = m.yes_token THEN f.price ELSE 1 - f.price END AS p_yes,
           f.size * f.price AS usd, m.outcome, m.resolve_ts + 60 - f.ts AS tau
    FROM fills f JOIN markets m USING (condition_id)
    WHERE m.outcome IS NOT NULL AND m.resolve_ts + 60 - f.ts BETWEEN 3600 AND 3*3600""").df()
d = d[(d.p_yes >= 0.35) & (d.p_yes <= 0.80)]
day = d.groupby("event_date").apply(lambda g: pd.Series(dict(
    n_fills=len(g), usd=g.usd.sum(), vw_price=np.average(g.p_yes, weights=g.usd),
    realized=np.average(g.outcome, weights=g.usd)))).reset_index()
day["gap"] = day.vw_price - day.realized
print(f"event-days={len(day)}  mean gap (price - realized, $-weighted)={day.gap.mean():+.3f}  "
      f"median={day.gap.median():+.3f}  SE={day.gap.std()/math.sqrt(len(day)):.3f}  "
      f"days overpriced={int((day.gap>0).sum())}/{len(day)}")
# same thing for 3-24h as a control (should show the drift sign, i.e. negative gap)
d2 = con.execute("""
    SELECT m.event_date, CASE WHEN f.token = m.yes_token THEN f.price ELSE 1 - f.price END AS p_yes,
           f.size * f.price AS usd, m.outcome
    FROM fills f JOIN markets m USING (condition_id)
    WHERE m.outcome IS NOT NULL AND m.resolve_ts + 60 - f.ts BETWEEN 3*3600 AND 24*3600""").df()
d2 = d2[(d2.p_yes >= 0.35) & (d2.p_yes <= 0.80)]
day2 = d2.groupby("event_date").apply(lambda g: np.average(g.p_yes, weights=g.usd) - np.average(g.outcome, weights=g.usd))
print(f"control 3-24h same price band: event-days={len(day2)} mean gap={day2.mean():+.3f} SE={day2.std()/math.sqrt(len(day2)):.3f}")

print("\n=== 3. post-print flow by minute after candle close (winner-token price, taker side) ===")
pp = con.execute("""
    SELECT (f.ts - (m.resolve_ts + 60)) / 60 AS min_after,
           CASE WHEN (f.token = m.yes_token) = (m.outcome = 1) THEN f.price ELSE 1 - f.price END AS winner_px,
           CASE WHEN (f.token = m.yes_token) = (m.outcome = 1) THEN f.side ELSE
                CASE WHEN f.side = 'BUY' THEN 'SELL' ELSE 'BUY' END END AS taker_side_on_winner,
           f.size * f.price AS usd
    FROM fills f JOIN markets m USING (condition_id)
    WHERE m.outcome IS NOT NULL AND f.ts >= m.resolve_ts + 60""").df()
pp["bucket"] = pd.cut(pp.min_after, [-1, 0, 1, 2, 5, 15, 60, 240, 1e9],
                      labels=["<1m", "1-2m", "2-3m", "3-5m", "5-15m", "15-60m", "1-4h", ">4h"])
g = pp.groupby(["bucket", "taker_side_on_winner"], observed=True).agg(
    n=("usd", "size"), usd=("usd", "sum"), winner_px=("winner_px", "mean"),
    px_lt_099=("winner_px", lambda s: (s < 0.99).mean())).reset_index()
g["usd"] = g.usd.round(0)
print(g.to_string(index=False))
days = con.execute("SELECT COUNT(DISTINCT event_date) FROM markets WHERE outcome IS NOT NULL").fetchone()[0]
sellers = pp[pp.taker_side_on_winner == "SELL"]
print(f"\nwinner-sellers post-print: ${sellers.usd.sum():,.0f} over {days} days = ${sellers.usd.sum()/days:,.0f}/day, "
      f"avg px {sellers.winner_px.mean():.4f} -> maker gross ≈ ${(sellers.usd*(1-sellers.winner_px)).sum()/days:,.2f}/day")
