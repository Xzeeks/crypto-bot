#!/usr/bin/env python3
"""c4_filtered_backtest.py — the filtered carry rule replayed on the 4 months of taker fills.

    python research/scripts/c4_filtered_backtest.py research/data/eth_above.duckdb research/data/census_eth.duckdb --pct 5 8
    python research/scripts/c4_filtered_backtest.py research/data/btc_above.duckdb research/data/census_btc.duckdb --pct 4 8

Rule: buy the favoured side (spot above strike -> Yes, spot below -> No) when
  * time to the noon print is in [--hours lo hi]           (default 12-24h)
  * |spot - strike| / strike in [--pct lo hi]              (default 5-8%)
  * the favoured-side price q <= --max-q                   (default 0.985)
  * filters: 'down' = downside strikes only (spot above strike; loss = a drop);
             'lowvol' = trailing-30d realized vol (noon-to-noon, from the census klines) below the 3-year tercile cut
Execution model: you are the counterparty to every qualifying taker fill at its price (i.e. you were the resting
order it hit), up to --cap $ of favoured-side notional per event-day, filled in time order. Report per-day P&L.
Ablation: none / lowvol / down / lowvol+down, so each filter's contribution is visible.
"""
import argparse, math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import duckdb, numpy as np, pandas as pd
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40)
NY = ZoneInfo("America/New_York")

ap = argparse.ArgumentParser()
ap.add_argument("fills_db"); ap.add_argument("census_db")
ap.add_argument("--pct", nargs=2, type=float, default=[5, 8])
ap.add_argument("--hours", nargs=2, type=float, default=[12, 24])
ap.add_argument("--max-q", type=float, default=0.985)
ap.add_argument("--cap", type=float, default=5000, help="max favoured-side $ per event-day")
a = ap.parse_args()

# --- realized-vol regime from the 3-year census klines ---
cz = duckdb.connect(a.census_db, read_only=True)
kl3 = cz.execute("SELECT open_ts, close FROM klines ORDER BY open_ts").df().set_index("open_ts")["close"]
d0 = datetime.fromtimestamp(kl3.index[0], NY).date() + timedelta(days=1); d1 = datetime.fromtimestamp(kl3.index[-1], NY).date()
rows, cur = [], d0
while cur <= d1:
    rts = int(datetime(cur.year, cur.month, cur.day, 12, 0, tzinfo=NY).timestamp())
    if (rts - 60) in kl3.index: rows.append((rts, kl3.loc[rts - 60]))
    cur += timedelta(days=1)
noon = pd.DataFrame(rows, columns=["rts", "print"]).set_index("rts")
noon["rv30"] = np.log(noon["print"]).diff().rolling(30).std() * math.sqrt(365) * 100
cut_lo = noon.rv30.quantile(1 / 3); cut_hi = noon.rv30.quantile(2 / 3)
# RV known at entry: use the value as of the PREVIOUS noon (no look-ahead on the day being traded)
noon["rv_prev"] = noon.rv30.shift(1)
print(f"3y RV30 terciles: low < {cut_lo:.0f}%  mid  high > {cut_hi:.0f}%")

# --- fills ---
con = duckdb.connect(a.fills_db, read_only=True)
kl = con.execute("SELECT open_ts, close FROM klines ORDER BY open_ts").df().set_index("open_ts")["close"]
f = con.execute("""
    SELECT f.ts, f.price, f.size, f.token, m.strike, m.resolve_ts, m.outcome, m.yes_token, m.event_date
    FROM fills f JOIN markets m USING (condition_id) WHERE m.outcome IS NOT NULL AND f.ts < m.resolve_ts""").df()
f["spot"] = ((f.ts // 60) * 60).map(kl); f = f.dropna(subset=["spot"])
f["hours"] = (f.resolve_ts - f.ts) / 3600
f["pct"] = (f.spot - f.strike).abs() / f.strike * 100
f["fav_yes"] = f.spot > f.strike
p_yes = np.where(f.token == f.yes_token, f.price, 1 - f.price)
f["q"] = np.where(f.fav_yes, p_yes, 1 - p_yes)
f["fav_won"] = np.where(f.fav_yes, f.outcome == 1, f.outcome == 0)
f["usd"] = f["size"] * f["q"]; f["pnl"] = f["size"] * (f.fav_won.astype(float) - f.q)
f["rv_prev"] = f.resolve_ts.map(noon.rv_prev)
base = f[(f.hours.between(*a.hours)) & (f.pct.between(*a.pct)) & (f.q <= a.max_q)].sort_values("ts")
all_days = sorted(f.resolve_ts.unique())

def run(sel, label):
    # per-day cap, in time order
    taken, spent = [], {}
    for r in sel.itertuples():
        s = spent.get(r.resolve_ts, 0.0)
        if s >= a.cap: continue
        frac = min(1.0, (a.cap - s) / r.usd) if r.usd > 0 else 0
        spent[r.resolve_ts] = s + r.usd * frac
        taken.append((r.resolve_ts, r.usd * frac, r.pnl * frac, r.fav_won, r.q))
    t = pd.DataFrame(taken, columns=["rts", "usd", "pnl", "won", "q"])
    if t.empty:
        print(f"{label:16s} no fills"); return
    d = t.groupby("rts").agg(usd=("usd", "sum"), pnl=("pnl", "sum"), losses=("won", lambda w: int((~w).sum()))).reindex(all_days, fill_value=0)
    eq = d.pnl.cumsum(); dd = (eq - eq.cummax()).min()
    print(f"{label:16s} days traded {int((d.usd>0).sum()):3d}/{len(d)}  deployed ${d.usd.sum():>9,.0f}  "
          f"avg/day ${d.usd[d.usd>0].mean():>7,.0f}  P&L ${d.pnl.sum():>+8,.0f}  ROI {d.pnl.sum()/d.usd.sum():+.2%}  "
          f"avg q {t.q.mean():.4f}  loss days {int((d.losses>0).sum())}  worst day ${d.pnl.min():>+8,.0f}  maxDD ${dd:>+8,.0f}")
    return d

print(f"\nrule: {a.hours[0]:.0f}-{a.hours[1]:.0f}h out, {a.pct[0]:.0f}-{a.pct[1]:.0f}% away, q <= {a.max_q}, cap ${a.cap:,.0f}/day, "
      f"{len(all_days)} event-days\n")
res = {}
res["none"] = run(base, "no filter")
res["lowvol"] = run(base[base.rv_prev < cut_lo], "lowvol")
res["down"] = run(base[base.fav_yes], "down only")
res["both"] = run(base[base.fav_yes & (base.rv_prev < cut_lo)], "lowvol+down")
res["up"] = run(base[~base.fav_yes], "(up only, ref)")
res["midvol"] = run(base[(base.rv_prev >= cut_lo) & (base.rv_prev < cut_hi)], "(midvol, ref)")
res["highvol"] = run(base[base.rv_prev >= cut_hi], "(highvol, ref)")

print("\nRV regime over the fills window:")
w = noon.loc[noon.index.isin(all_days), ["rv30"]]
print(f"  days low {int((w.rv30<cut_lo).sum())}, mid {int(((w.rv30>=cut_lo)&(w.rv30<cut_hi)).sum())}, high {int((w.rv30>=cut_hi).sum())}; "
      f"RV30 range {w.rv30.min():.0f}-{w.rv30.max():.0f}%")

if res.get("both") is not None:
    d = res["both"]; dl = d[d.losses > 0]
    print("\nlowvol+down loss days:")
    print(pd.DataFrame({"date": [datetime.fromtimestamp(x, NY).date() for x in dl.index], "deployed": dl.usd.round(0),
                        "pnl": dl.pnl.round(0), "rv_prev": [noon.rv_prev.get(x, np.nan) for x in dl.index]}).round(1).to_string(index=False))
    print("\nlowvol+down daily P&L distribution (traded days): " +
          ", ".join(f"p{p} ${d.pnl[d.usd>0].quantile(p/100):,.0f}" for p in (5, 25, 50, 75, 95)))
