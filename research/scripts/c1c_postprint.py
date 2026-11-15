#!/usr/bin/env python3
"""c1c_postprint.py — post-print flow on the daily 'above $X' ladder, with the corrected print time.

    python research/scripts/c1c_postprint.py research/data/eth_above.duckdb [research/data/btc_above.duckdb ...]

Print time = markets.resolve_ts (12:00:00 ET). The resolving candle OPENS at resolve_ts - 60 and CLOSES at resolve_ts.
A fill at ts >= resolve_ts happens with the outcome knowable from Binance.

For each resolved strike:
  winner_px        = price of the winning token on the fill (Yes px if Yes won, else 1 - Yes px)
  taker_on_winner  = BUY if the taker bought the winner, SELL if the taker sold it
  maker_gross      = usd * (1 - winner_px) on taker-SELL fills   (resting bid collects this)
  taker_gross      = usd * (1 - winner_px) on taker-BUY fills    (lifting a stale ask collects this)
Reports: seconds-after buckets, per-day pool distribution, wallet concentration, near-strike sensitivity,
and last pre-print winner price (what the winner was trading at in the final minute).
"""
import sys
import duckdb, numpy as np, pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

for db in sys.argv[1:]:
    con = duckdb.connect(db, read_only=True)
    print(f"\n##### {db}")
    tot, bad = con.execute("""
        SELECT COUNT(*), SUM(CASE WHEN (k.close > m.strike) <> (m.outcome = 1) THEN 1 ELSE 0 END)
        FROM markets m JOIN klines k ON k.open_ts = m.resolve_ts - 60 WHERE m.outcome IS NOT NULL""").fetchone()
    print(f"rule check (candle closing at 12:00 ET): {tot} resolved, {bad} mismatches")
    nf = con.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    if nf == 0:
        print("no fills in this DB (run the pull without --skip-fills)"); continue

    pp = con.execute("""
        SELECT m.event_date, m.strike, m.outcome, f.ts - m.resolve_ts AS s_after, f.wallet, f.size * f.price AS usd,
               CASE WHEN (f.token = m.yes_token) = (m.outcome = 1) THEN f.price ELSE 1 - f.price END AS winner_px,
               CASE WHEN (f.token = m.yes_token) = (m.outcome = 1) THEN f.side
                    ELSE CASE WHEN f.side = 'BUY' THEN 'SELL' ELSE 'BUY' END END AS taker_on_winner,
               ABS(k.close - m.strike) AS dist_to_strike
        FROM fills f JOIN markets m USING (condition_id)
        JOIN klines k ON k.open_ts = m.resolve_ts - 60
        WHERE m.outcome IS NOT NULL AND f.ts >= m.resolve_ts""").df()
    days = con.execute("SELECT COUNT(DISTINCT event_date) FROM markets WHERE outcome IS NOT NULL").fetchone()[0]
    pp["gross"] = pp.usd * (1 - pp.winner_px)
    pp["bucket"] = pd.cut(pp.s_after, [-1, 10, 30, 60, 120, 300, 900, 3600, 4*3600, 1e9],
                          labels=["0-10s", "10-30s", "30-60s", "1-2m", "2-5m", "5-15m", "15-60m", "1-4h", ">4h"])

    print(f"\n=== flow by seconds after print ({days} event-days) ===")
    g = pp.groupby(["bucket", "taker_on_winner"], observed=True).agg(
        n=("usd", "size"), usd=("usd", "sum"), px=("winner_px", "mean"),
        vw_px=("winner_px", lambda s: np.average(s, weights=pp.loc[s.index, "usd"])),
        gross=("gross", "sum"), lt_099=("winner_px", lambda s: (s < 0.99).mean())).reset_index()
    g["usd"] = g.usd.round(0); g["gross"] = g.gross.round(0)
    print(g.to_string(index=False))

    for side, who in [("SELL", "maker (resting bid)"), ("BUY", "taker (lifting stale ask)")]:
        s = pp[pp.taker_on_winner == side]
        byday = s.groupby("event_date").agg(usd=("usd", "sum"), gross=("gross", "sum")).reindex(
            con.execute("SELECT DISTINCT event_date FROM markets WHERE outcome IS NOT NULL").df().event_date, fill_value=0)
        print(f"\n=== {who}: taker-{side} fills on the winner ===")
        print(f"total ${s.usd.sum():,.0f}, gross ${s.gross.sum():,.0f} over {days} days = ${s.gross.sum()/days:,.0f}/day")
        print(f"per-day gross: p10 ${byday.gross.quantile(.1):,.0f}  p25 ${byday.gross.quantile(.25):,.0f}  "
              f"median ${byday.gross.median():,.0f}  p75 ${byday.gross.quantile(.75):,.0f}  max ${byday.gross.max():,.0f}  "
              f"days with any flow {int((byday.usd>0).sum())}/{days}")
        print(f"excluding near-strike days (<$5): gross ${s[s.dist_to_strike>=5].gross.sum():,.0f}; "
              f"excluding fills at px<0.90: gross ${s[s.winner_px>=0.90].gross.sum():,.0f}")
        w = s.groupby("wallet").agg(usd=("usd", "sum"), gross=("gross", "sum"), n=("usd", "size")).sort_values("gross", ascending=False)
        print(f"distinct taker wallets {len(w)}; top-5 share of gross {w.gross.head(5).sum()/max(w.gross.sum(),1e-9):.0%}")
        print(w.head(5).round(0).to_string())
        sz = pd.cut(s.usd, [0, 10, 50, 200, 1000, 5000, 1e9])
        print("fill size distribution ($):"); print(s.groupby(sz, observed=True).agg(n=("usd", "size"), usd=("usd", "sum"), gross=("gross", "sum")).round(0).to_string())

    print("\n=== winner price in the last 60s BEFORE the print (how much was already known) ===")
    pre = con.execute("""
        SELECT CASE WHEN (f.token = m.yes_token) = (m.outcome = 1) THEN f.price ELSE 1 - f.price END AS winner_px,
               f.size * f.price AS usd
        FROM fills f JOIN markets m USING (condition_id)
        WHERE m.outcome IS NOT NULL AND f.ts >= m.resolve_ts - 60 AND f.ts < m.resolve_ts""").df()
    if len(pre):
        print(pre.groupby(pd.cut(pre.winner_px, [0, .5, .9, .95, .98, .99, .995, 1.0001]), observed=True)
              .agg(n=("usd", "size"), usd=("usd", "sum")).round(0).to_string())
    con.close()
