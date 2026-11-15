#!/usr/bin/env python3
"""c8_touch_grid.py — the one-touch version of the sweet-spot grid.

    python research/scripts/c8_touch_grid.py research/data/eth_hit.duckdb research/data/census_eth.duckdb
    python research/scripts/c8_touch_grid.py research/data/btc_hit.duckdb research/data/census_btc.duckdb --step-pct 2.4

Trade = buy NO on "↓ X" (ETH stays above X all day) or NO on "↑ X" (stays below). Loss = the barrier is touched
(any 1m low <= X, or high >= X) between entry and 23:59 ET.

Table 1 (3-year klines from the census DB, no Polymarket): for every day, every lookback H before midnight ET, every
  barrier on a %-grid: was it touched before the day ended?  -> true touch rate per (hours-left x %-away x direction).
  (Entry at H hours before midnight; a barrier already touched earlier that day is skipped — you wouldn't sell it.)
Table 2 (fills): every taker fill restated as buying NO at price q = 1 - p_yes, same cells: n, $, avg q, realized win,
  ROI, vs Table 1's true rate -> breakeven comparison.
Table 3: same-day comparison with the close-based ladder if --above DB is given: for strikes present in both, the
  touch NO price vs the close Yes price at the same minute (how much extra the market charges for path risk).
"""
import argparse, math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import duckdb, numpy as np, pandas as pd
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
NY = ZoneInfo("America/New_York")

ap = argparse.ArgumentParser()
ap.add_argument("hit_db"); ap.add_argument("census_db")
ap.add_argument("--step-pct", type=float, default=1.0, help="barrier grid step as % of spot for Table 1")
ap.add_argument("--above", default=None, help="optional close-ladder duckdb (c1) for Table 3")
a = ap.parse_args()

PCT = [0, 1, 2, 3, 4, 5, 7, 10, 100]
HRS = [0, 1, 2, 3, 6, 9, 12, 18, 24, 1e9]; HRS_LBL = ["<1h", "1-2h", "2-3h", "3-6h", "6-9h", "9-12h", "12-18h", "18-24h", ">24h"]

# ---------- Table 1: 3-year touch census ----------
cz = duckdb.connect(a.census_db, read_only=True)
k = cz.execute("SELECT open_ts, high, low, close FROM klines ORDER BY open_ts").df().set_index("open_ts")
d0 = datetime.fromtimestamp(k.index[0], NY).date() + timedelta(days=1); d1 = datetime.fromtimestamp(k.index[-1], NY).date() - timedelta(days=1)
rows, cur = [], d0
hi_arr, lo_arr, cl = k.high, k.low, k.close
while cur <= d1:
    t0 = int(datetime(cur.year, cur.month, cur.day, tzinfo=NY).timestamp()); t1 = t0 + 86400
    day = k.loc[t0:t1 - 60]
    if len(day) < 1300: cur += timedelta(days=1); continue
    for h in (0.5, 1, 1.5, 2, 3, 4.5, 6, 7.5, 9, 10.5, 12, 15, 18, 21):
        te = t1 - int(h * 3600)
        if te not in day.index: continue
        spot = day.loc[te, "close"]; rest = day.loc[te + 60:]
        past = day.loc[:te]
        rmax, rmin = rest.high.max(), rest.low.min(); pmax, pmin = past.high.max(), past.low.min()
        step = spot * a.step_pct / 100
        for i in range(1, int(12 / a.step_pct) + 1):
            kd = spot - i * step; ku = spot + i * step
            if pmin > kd: rows.append((h, i * a.step_pct, "↓", rmin <= kd))      # not yet touched today -> sellable
            if pmax < ku: rows.append((h, i * a.step_pct, "↑", rmax >= ku))
    cur += timedelta(days=1)
t1 = pd.DataFrame(rows, columns=["hours", "pct", "dir", "touched"])
t1["pct_b"] = pd.cut(t1.pct, PCT); t1["h_b"] = pd.cut(t1.hours, HRS, labels=HRS_LBL)
g1 = t1.groupby(["dir", "h_b", "pct_b"], observed=True).agg(n=("touched", "size"), touched=("touched", "sum"))
g1["touch_pct"] = (g1.touched / g1.n * 100).round(2)
print(f"=== Table 1: 3-year touch rate % (barrier not yet touched at entry; entry H hours before midnight ET) ===")
for d_ in ("↓", "↑"):
    print(f"\n-- {d_} barriers ({'loss = drop touches' if d_ == '↓' else 'loss = rally touches'}) --")
    print(g1.loc[d_].touch_pct.unstack("pct_b").to_string())
print("\nbreakeven NO price = 1 - touch rate (↓):"); print((1 - g1.loc["↓"].touched / g1.loc["↓"].n).unstack("pct_b").round(4).to_string())

# ---------- Table 2: fills ----------
con = duckdb.connect(a.hit_db, read_only=True)
kl = con.execute("SELECT open_ts, high, low, close FROM klines ORDER BY open_ts").df().set_index("open_ts")
f = con.execute("""SELECT f.ts, f.price, f.size, f.token, f.side, m.event_date, m.direction, m.strike, m.window_start, m.window_end, m.outcome, m.yes_token
                   FROM fills f JOIN markets m USING (condition_id) WHERE m.outcome IS NOT NULL AND f.ts >= m.window_start AND f.ts < m.window_end""").df()
if f.empty: raise SystemExit("no fills")
f["minute"] = (f.ts // 60) * 60
f["spot"] = f.minute.map(kl.close); f = f.dropna(subset=["spot"])
# already touched before the fill? (then NO is dead — exclude; you'd never sell it)
def touched_before(r):
    seg = kl.loc[r.window_start:r.minute - 60]
    if seg.empty: return False
    return (seg.low.min() <= r.strike) if r.direction == "↓" else (seg.high.max() >= r.strike)
f["dead"] = [touched_before(r) for r in f.itertuples()]
f = f[~f.dead]
f["hours"] = (f.window_end - f.ts) / 3600
f["pct"] = np.where(f.direction == "↓", (f.spot - f.strike) / f.spot * 100, (f.strike - f.spot) / f.spot * 100)
f = f[f.pct > 0]
p_yes = np.where(f.token == f.yes_token, f.price, 1 - f.price)
f["q"] = 1 - p_yes                               # price of NO
f["no_won"] = f.outcome == 0
f["usd"] = f["size"] * f.q; f["pnl"] = f["size"] * (f.no_won.astype(float) - f.q)
f["pct_b"] = pd.cut(f.pct, PCT); f["h_b"] = pd.cut(f.hours, HRS, labels=HRS_LBL)
g2 = f.groupby(["direction", "h_b", "pct_b"], observed=True).agg(n=("q", "size"), usd=("usd", "sum"), avg_q=("q", "mean"), p10_q=("q", lambda s: s.quantile(.1)),
                                                                   win=("no_won", "mean"), losses=("no_won", lambda s: int((~s).sum())), pnl=("pnl", "sum")).reset_index()
g2["roi"] = g2.pnl / g2.usd
g2 = g2.merge(g1.reset_index().rename(columns={"dir": "direction"})[["direction", "h_b", "pct_b", "touch_pct"]], on=["direction", "h_b", "pct_b"], how="left")
g2["breakeven_q"] = 1 - g2.touch_pct / 100
g2["usd"] = g2.usd.round(0); g2["pnl"] = g2.pnl.round(0)
days = f.event_date.nunique()
print(f"\n=== Table 2: buying NO on live barriers ({days} event-days, {len(f):,} fills) — avg_q vs breakeven_q is the edge ===")
print(g2[g2.n >= 30].round(4).to_string(index=False))

# ---------- Table 3: touch NO vs close Yes, same strike & minute ----------
if a.above:
    ab = duckdb.connect(a.above, read_only=True)
    g = ab.execute("""SELECT m.event_date, m.strike, (f.ts // 60) * 60 AS minute,
                             AVG(CASE WHEN f.token = m.yes_token THEN f.price ELSE 1 - f.price END) AS close_yes
                      FROM fills f JOIN markets m USING (condition_id) WHERE m.outcome IS NOT NULL GROUP BY 1,2,3""").df()
    g["event_date"] = g.event_date.astype(str)
    h = f[f.direction == "↓"].groupby(["event_date", "strike", "minute"]).q.mean().reset_index().rename(columns={"q": "touch_no"})
    h["event_date"] = h.event_date.astype(str)
    j = h.merge(g, on=["event_date", "strike", "minute"])
    j["path_premium"] = j.close_yes - j.touch_no
    print(f"\n=== Table 3: same strike, same minute: close-ladder Yes vs touch-ladder NO ({len(j):,} matched minutes) ===")
    print("NB: close ladder resolves 12:00 ET, touch ladder runs to 23:59 ET on the same date, so horizons differ by 12h — read as indicative.")
    jj = j.merge(f[["event_date", "strike", "minute", "pct"]].drop_duplicates(), on=["event_date", "strike", "minute"])
    jj["pct_b"] = pd.cut(jj.pct, PCT)
    print(jj.groupby("pct_b", observed=True).agg(n=("path_premium", "size"), close_yes=("close_yes", "mean"), touch_no=("touch_no", "mean"), premium=("path_premium", "mean")).round(4).to_string())
