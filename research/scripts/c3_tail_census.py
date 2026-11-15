#!/usr/bin/env python3
"""c3_tail_census.py — multi-year loss-rate census for the "X% away, H hours before noon ET" carry band.

    python research/scripts/c3_tail_census.py --symbol ETHUSDT --years 3 --out research/data/census_eth.duckdb
    python research/scripts/c3_tail_census.py --symbol BTCUSDT --years 3 --out research/data/census_btc.duckdb --step-pct 2.4

Pulls 1m klines from data-api.binance.vision (resumable), then for every calendar day: the resolving print is the
close of the 1m candle that CLOSES at 12:00 America/New_York. For each lookback H and each strike on a grid around
spot-at-H, did the favoured side (spot above strike -> Yes, else No) survive to the print?

The strike grid is a percentage step (ETH's $100 grid is ~4% of a $2.5k spot; BTC's $2000 is ~2.4% of $84k) so
the census stays comparable as the price level changes over 3 years; cells are bucketed by % distance anyway.

Reports per (hours x %-away) cell: n, losses, loss rate, breakeven price (1 - loss rate), and the loss rate split by
year, by direction (upside strike = Yes favoured? no: favoured side is No -> a DOWN-tail loss ... see labels) and by
trailing-30d realized-vol tercile. Also: per-day worst move so you can see the fat-tail days.
"""
import argparse, math, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import duckdb, numpy as np, pandas as pd, requests

pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
NY = ZoneInfo("America/New_York")
BINANCE = "https://data-api.binance.vision/api/v3/klines"
S = requests.Session(); S.headers["User-Agent"] = "c3-tail-census/0.1"

ap = argparse.ArgumentParser()
ap.add_argument("--symbol", default="ETHUSDT")
ap.add_argument("--years", type=float, default=3.0)
ap.add_argument("--out", required=True)
ap.add_argument("--step-pct", type=float, default=4.0, help="strike grid step as %% of spot (ETH ~4, BTC ~2.4)")
ap.add_argument("--skip-pull", action="store_true")
a = ap.parse_args()

con = duckdb.connect(a.out)
con.execute("CREATE TABLE IF NOT EXISTS klines(open_ts BIGINT PRIMARY KEY, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE)")

# ---------- pull ----------
if not a.skip_pull:
    end_ts = int(time.time()) // 60 * 60
    start_ts = end_ts - int(a.years * 365.25 * 86400)
    have = con.execute("SELECT MAX(open_ts) FROM klines").fetchone()[0]
    cur = (max(start_ts, have + 60) if have else start_ts) * 1000
    n = 0
    while cur < end_ts * 1000:
        for i in range(5):
            try:
                r = S.get(BINANCE, params={"symbol": a.symbol, "interval": "1m", "startTime": cur, "endTime": end_ts * 1000, "limit": 1000}, timeout=30)
                if r.status_code == 429: time.sleep(5 + 5 * i); continue
                r.raise_for_status(); b = r.json(); break
            except Exception as e:
                if i == 4: raise
                time.sleep(2 + 2 * i)
        if not b: break
        con.executemany("INSERT OR REPLACE INTO klines VALUES (?,?,?,?,?,?)",
                        [(k[0] // 1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in b])
        n += len(b); cur = b[-1][0] + 60_000
        if n % 50000 < 1000: print(f"  klines {n:,} up to {datetime.fromtimestamp(cur/1000, timezone.utc):%Y-%m-%d}", flush=True)
        if len(b) < 1000: break
    print(f"pulled {n:,} new klines")

kl = con.execute("SELECT open_ts, close FROM klines ORDER BY open_ts").df().set_index("open_ts")["close"]
print(f"klines: {len(kl):,} rows, {datetime.fromtimestamp(kl.index[0], timezone.utc):%Y-%m-%d} .. {datetime.fromtimestamp(kl.index[-1], timezone.utc):%Y-%m-%d}")

# trailing 30d realized vol (daily, from the noon-to-noon log returns), for regime split
d0 = datetime.fromtimestamp(kl.index[0], NY).date() + timedelta(days=1)
d1 = datetime.fromtimestamp(kl.index[-1], NY).date()
days = []
cur = d0
while cur <= d1:
    rts = int(datetime(cur.year, cur.month, cur.day, 12, 0, tzinfo=NY).timestamp())
    if (rts - 60) in kl.index: days.append((cur, rts, kl.loc[rts - 60]))
    cur += timedelta(days=1)
noon = pd.DataFrame(days, columns=["date", "rts", "print"]).set_index("date")
noon["ret"] = np.log(noon["print"]).diff()
noon["rv30"] = noon["ret"].rolling(30).std() * math.sqrt(365) * 100   # annualized %
noon["rv_tercile"] = pd.qcut(noon["rv30"], 3, labels=["low", "mid", "high"])

# ---------- census ----------
PCT = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 100]
HRS = [0, 1, 3, 6, 9, 12, 18, 24, 48, 1e9]
HRS_LBL = ["<1h", "1-3h", "3-6h", "6-9h", "9-12h", "12-18h", "18-24h", "1-2d", ">2d"]
lookbacks = [0.5, 1, 2, 3, 4.5, 6, 7.5, 9, 10.5, 12, 15, 18, 21, 24, 36, 48]
rows = []
for date, r in noon.iterrows():
    rts, final = r.rts, r["print"]
    for h in lookbacks:
        t = rts - int(h * 3600)
        if t not in kl.index: continue
        spot = kl.loc[t]
        step = spot * a.step_pct / 100
        base = math.floor(spot / step) * step
        for k_i in range(-4, 5):   # strikes from ~-16% to +16% of spot (ETH ladder is 11 wide; BTC 11 x 2.4%)
            k = base + k_i * step
            if k <= 0 or abs(spot - k) < 1e-9: continue
            fav_yes = spot > k
            won = (final > k) == fav_yes
            rows.append((date, date.year, h, abs(spot - k) / k * 100, fav_yes, won, r.rv_tercile, (final / spot - 1) * 100))
c = pd.DataFrame(rows, columns=["date", "year", "hours", "pct", "fav_yes", "won", "rv", "move_pct"])
c["pct_b"] = pd.cut(c.pct, PCT); c["h_b"] = pd.cut(c.hours, HRS, labels=HRS_LBL)
c["direction"] = np.where(c.fav_yes, "spot ABOVE strike (loss = drop)", "spot BELOW strike (loss = rally)")

def table(df, title):
    g = df.groupby(["h_b", "pct_b"], observed=True).agg(n=("won", "size"), losses=("won", lambda s: int((~s).sum())))
    g["loss_pct"] = (g.losses / g.n * 100).round(2)
    print(f"\n=== {title}: loss rate % of the favoured side ===")
    print(g.loss_pct.unstack("pct_b").to_string())
    return g

g_all = table(c, f"ALL ({c.date.nunique()} days, {a.symbol})")
print("\nn per cell:"); print(g_all.n.unstack("pct_b").to_string())
print("\nbreakeven price (1 - loss rate) for the favoured side:")
print(((1 - g_all.losses / g_all.n)).unstack("pct_b").round(4).to_string())

for y, sub in c.groupby("year"):
    table(sub, f"year {y} ({sub.date.nunique()} days)")
for d_, sub in c.groupby("direction"):
    table(sub, d_)
for t_, sub in c.groupby("rv", observed=True):
    rv_rng = noon[noon.rv_tercile == t_].rv30
    table(sub, f"trailing-30d realized vol tercile '{t_}' ({rv_rng.min():.0f}-{rv_rng.max():.0f}% ann., {sub.date.nunique()} days)")

# ---------- the band in focus, by year, in one line each ----------
print("\n=== focus band (12-24h out) by year: loss rate and breakeven for 3-4 / 4-5 / 5-7 / 7-10 % away ===")
foc = c[c.h_b.isin(["12-18h", "18-24h"])]
piv = foc.groupby(["year", "pct_b"], observed=True).agg(n=("won", "size"), losses=("won", lambda s: int((~s).sum())))
piv["loss_pct"] = (piv.losses / piv.n * 100).round(2)
pv = piv.loss_pct.unstack("pct_b")
print(pv[[col for col in pv.columns if col.left in (3, 4, 5, 7)]].to_string())

print("\n=== biggest noon-to-noon moves (the days that pay the tail) ===")
big = noon["ret"].abs().sort_values(ascending=False).head(15)
print(pd.DataFrame({"move_pct": (np.exp(noon.loc[big.index, "ret"]) - 1) * 100, "rv30": noon.loc[big.index, "rv30"]}).round(2).to_string())
print(f"\ndays with |noon-to-noon move| > 5%: {int((noon.ret.abs() > math.log(1.05)).sum())} / {len(noon)}   > 7%: "
      f"{int((noon.ret.abs() > math.log(1.07)).sum())}   > 10%: {int((noon.ret.abs() > math.log(1.10)).sum())}")
con.close()
