#!/usr/bin/env python3
"""c14_klines_backfill.py — extend the census price history BACKWARDS (c3 only ever pulls forward from what it has).

    python research/scripts/c14_klines_backfill.py --symbol ETHUSDT --out research/data/census_eth.duckdb [--start 2020-01-01]
    python research/scripts/c14_klines_backfill.py --symbol BTCUSDT --out research/data/census_btc.duckdb [--start 2020-01-01]

Pulls 1m klines from data-api.binance.vision for [--start, first kline already in the DB) and inserts them into the same
`klines` table, so c12_conditioning.py picks the longer history up with no change. Resumable: the original first kline is
remembered in a `backfill` table, and a rerun continues from the last older kline it stored. Rerunning when complete is a no-op.
"""
import argparse, time
from datetime import datetime, timezone
import duckdb, pandas as pd, requests

BINANCE = "https://data-api.binance.vision/api/v3/klines"
ap = argparse.ArgumentParser()
ap.add_argument("--symbol", default="ETHUSDT"); ap.add_argument("--out", required=True)
ap.add_argument("--start", default="2020-01-01", help="UTC date to backfill from")
a = ap.parse_args()
S = requests.Session(); S.headers["User-Agent"] = "c14-klines-backfill/0.1"
utc = lambda ts: datetime.fromtimestamp(ts, timezone.utc)

con = duckdb.connect(a.out)
if not con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'klines'").fetchone()[0]:
    raise SystemExit(f"{a.out} has no klines table — run c3_tail_census.py first")
con.execute("CREATE TABLE IF NOT EXISTS backfill(symbol TEXT PRIMARY KEY, boundary_ts BIGINT)")
row = con.execute("SELECT boundary_ts FROM backfill WHERE symbol = ?", [a.symbol]).fetchone()
if row: boundary = row[0]
else:
    boundary = con.execute("SELECT MIN(open_ts) FROM klines").fetchone()[0]
    if boundary is None: raise SystemExit("klines table is empty — run c3_tail_census.py first")
    con.execute("INSERT INTO backfill VALUES (?, ?)", [a.symbol, boundary])
start_ts = int(datetime.strptime(a.start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
have = con.execute("SELECT MAX(open_ts) FROM klines WHERE open_ts < ?", [boundary]).fetchone()[0]
cur = max(start_ts, have + 60) if have is not None else start_ts
print(f"{a.symbol}: history starts {utc(boundary):%Y-%m-%d}; backfilling {utc(cur):%Y-%m-%d %H:%M} .. {utc(boundary):%Y-%m-%d %H:%M} UTC", flush=True)

n, nxt = 0, 100_000
while cur < boundary:
    for i in range(6):
        try:
            r = S.get(BINANCE, params={"symbol": a.symbol, "interval": "1m", "startTime": cur * 1000, "endTime": (boundary - 60) * 1000, "limit": 1000}, timeout=30)
            if r.status_code in (418, 429): time.sleep(5 + 5 * i); continue
            r.raise_for_status(); b = r.json(); break
        except Exception:
            if i == 5: raise
            time.sleep(2 + 2 * i)
    else:
        raise SystemExit("rate-limited repeatedly — wait a few minutes and rerun; it resumes where it stopped")
    if not b: break
    df = pd.DataFrame([(k[0] // 1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in b],
                      columns=["open_ts", "open", "high", "low", "close", "volume"])
    df = df[df.open_ts < boundary]
    con.execute("INSERT OR REPLACE INTO klines SELECT open_ts, open, high, low, close, volume FROM df")
    n += len(df); cur = b[-1][0] // 1000 + 60
    if n >= nxt: print(f"  klines {n:,} up to {utc(cur):%Y-%m-%d}", flush=True); nxt += 100_000
    if len(b) < 1000: break
lo, hi, tot = con.execute("SELECT MIN(open_ts), MAX(open_ts), COUNT(*) FROM klines").fetchone()
gap = con.execute("SELECT COUNT(*) FROM (SELECT open_ts - LAG(open_ts) OVER (ORDER BY open_ts) AS d FROM klines) WHERE d > 3600").fetchone()[0]
print(f"added {n:,} klines; table now {tot:,} rows, {utc(lo):%Y-%m-%d} .. {utc(hi):%Y-%m-%d}; gaps longer than an hour: {gap}")
con.close()
