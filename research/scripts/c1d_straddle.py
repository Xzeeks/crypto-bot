#!/usr/bin/env python3
"""c1d_straddle.py — the candle-convention edge: days where the 11:59-open candle (the resolving one) and the
12:00-open candle sit on opposite sides of a strike, and what the market did in the minutes after the print.

    python research/scripts/c1d_straddle.py research/data/eth_above.duckdb research/data/btc_above.duckdb

For every resolved strike: dist = close_1159 - strike (signed, resolving candle), straddle = sign differs from
close_1200 - strike. Then for near-strike days (|dist| within $X) list post-print fills (first 10 min) on the
TRUE winner below 0.80 — that is the confused flow — split by straddle vs non-straddle, plus the all-day
'mispriced winner' pool (any fill on the true winner below 0.90 after the print).
"""
import sys
import duckdb, pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

for db in sys.argv[1:]:
    con = duckdb.connect(db, read_only=True)
    print(f"\n##### {db}")
    near = con.execute("""
        SELECT m.event_date, m.strike, m.outcome, k0.close AS close_1159, k1.close AS close_1200,
               k0.close - m.strike AS dist, k0.low AS low_1159, k0.high AS high_1159,
               ((k0.close > m.strike) <> (k1.close > m.strike)) AS straddle,
               ((k0.close > m.strike) <> (m.outcome = 1)) AS rule_mismatch
        FROM markets m JOIN klines k0 ON k0.open_ts = m.resolve_ts - 60 JOIN klines k1 ON k1.open_ts = m.resolve_ts
        WHERE m.outcome IS NOT NULL ORDER BY ABS(k0.close - m.strike)""").df()
    n_days = near.event_date.nunique()
    for thr in (5, 10, 25, 50):
        sub = near[near.dist.abs() < thr]
        print(f"|close-strike| < ${thr}: {len(sub)} strike-days, straddles {int(sub.straddle.sum())}, "
              f"rule mismatches {int(sub.rule_mismatch.sum())}  (of {n_days} days)")
    print("\nclosest 15 strike-days:")
    print(near.head(15).to_string(index=False))

    print("\n=== post-print fills on the TRUE winner below 0.80, first 10 min, near-strike days (<$50) ===")
    conf = con.execute("""
        SELECT m.event_date, m.strike, f.ts - m.resolve_ts AS s_after, f.side AS taker_side_raw,
               CASE WHEN (f.token = m.yes_token) = (m.outcome = 1) THEN f.price ELSE 1 - f.price END AS winner_px,
               CASE WHEN (f.token = m.yes_token) = (m.outcome = 1) THEN f.side
                    ELSE CASE WHEN f.side='BUY' THEN 'SELL' ELSE 'BUY' END END AS taker_on_winner,
               f.size, f.size * f.price AS usd, f.wallet,
               ((k0.close > m.strike) <> (k1.close > m.strike)) AS straddle, k0.close - m.strike AS dist
        FROM fills f JOIN markets m USING (condition_id)
        JOIN klines k0 ON k0.open_ts = m.resolve_ts - 60 JOIN klines k1 ON k1.open_ts = m.resolve_ts
        WHERE m.outcome IS NOT NULL AND f.ts >= m.resolve_ts AND f.ts < m.resolve_ts + 600
          AND ABS(k0.close - m.strike) < 50
        ORDER BY m.event_date, f.ts""").df()
    conf = conf[conf.winner_px < 0.80].copy()
    conf["gross_to_counterparty"] = conf["size"] * (1 - conf.winner_px)
    if conf.empty:
        print("none")
    else:
        print(conf.groupby(["event_date", "strike", "straddle", "taker_on_winner"]).agg(
            n=("usd", "size"), usd=("usd", "sum"), shares=("size", "sum"), vw_px=("winner_px", "mean"),
            first_s=("s_after", "min"), last_s=("s_after", "max"), gross=("gross_to_counterparty", "sum"),
            wallets=("wallet", "nunique")).round(2).to_string())
        print("\nby straddle flag (taker SOLD the true winner = confused seller, maker side collects):")
        s = conf[conf.taker_on_winner == "SELL"]
        print(s.groupby("straddle").agg(n=("usd", "size"), usd=("usd", "sum"), gross=("gross_to_counterparty", "sum")).round(0).to_string())

    print("\n=== all post-print fills on the TRUE winner below 0.90 (any day, any time) — the whole pool ===")
    pool = con.execute("""
        SELECT m.event_date, m.strike, MIN(f.ts - m.resolve_ts) AS first_s, MAX(f.ts - m.resolve_ts) AS last_s,
               COUNT(*) AS n, SUM(f.size) AS shares, SUM(f.size * f.price) AS usd,
               SUM(f.size * (1 - CASE WHEN (f.token = m.yes_token) = (m.outcome = 1) THEN f.price ELSE 1 - f.price END)) AS gross,
               k0.close - m.strike AS dist
        FROM fills f JOIN markets m USING (condition_id) JOIN klines k0 ON k0.open_ts = m.resolve_ts - 60
        WHERE m.outcome IS NOT NULL AND f.ts >= m.resolve_ts
          AND (CASE WHEN (f.token = m.yes_token) = (m.outcome = 1) THEN f.price ELSE 1 - f.price END) < 0.90
        GROUP BY 1,2,dist ORDER BY gross DESC""").df()
    print(pool.head(20).round(2).to_string(index=False))
    con.close()
