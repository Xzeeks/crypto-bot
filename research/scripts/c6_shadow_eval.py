#!/usr/bin/env python3
"""c6_shadow_eval.py — score the close-ladder downside carry against logged order books (logger/book_logger.py, family='above').

    python research/scripts/c6_shadow_eval.py research/data/books.sqlite [--asset ethereum] [--fills research/data/eth_above.duckdb]
                                              [--pct 5 8] [--hours 12 24] [--max-q 0.975] [--cap 5000]

Rule: spot is 5-8% ABOVE a strike, 12-24h before the noon ET print -> buy Yes on that strike at <= max_q.
  QUEUE  where a resting Yes bid at max_q would sit in the logged book.
  TAKER  paper: at the FIRST qualifying snapshot of each strike-day, walk the ask levels at <= max_q (true VWAP),
         up to --cap $ per event-day.
  MAKER  (needs --fills): taker SELLS of Yes that hit resting bids at <= max_q on the cell's strikes INSIDE the cell's own
         time window (first to last qualifying snapshot of that strike-day), with the P&L of having been the maker on every
         print at its price. A ceiling for a maker; QUEUE says how often we would have been first in line.
Outcome: fills DB markets table when resolved; otherwise provisional from logged spot at the print (flagged).
"""
import argparse, json, sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
NY = ZoneInfo("America/New_York"); EPS = 1e-9

ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("--asset", default="ethereum")
ap.add_argument("--pct", nargs=2, type=float, default=[5, 8]); ap.add_argument("--hours", nargs=2, type=float, default=[12, 24])
ap.add_argument("--max-q", type=float, default=0.975); ap.add_argument("--cap", type=float, default=5000)
ap.add_argument("--fills", default=None, help="duckdb from the c1 pull: resolved outcomes + taker prints")
a = ap.parse_args()

con = sqlite3.connect(a.db)
b = pd.read_sql("SELECT * FROM books WHERE family='above' AND asset=? AND spot IS NOT NULL", con, params=(a.asset,))
if b.empty: raise SystemExit("no above-family books logged yet for " + a.asset)
b["hours"] = (b.resolve_ts - b.ts) / 3600
b["pct"] = (b.spot - b.strike) / b.strike * 100           # positive = spot above strike = we want Yes

out = {}
if a.fills:
    import duckdb
    dd = duckdb.connect(a.fills, read_only=True)
    for ed, k, o in dd.execute("SELECT event_date, strike, outcome FROM markets WHERE outcome IS NOT NULL").fetchall():
        out[(str(ed)[:10], float(k))] = int(o)
prov = {}
for (ed, k), g in b[b.ts <= b.resolve_ts].groupby(["event_date", "strike"]):
    last = g.loc[g.ts.idxmax()]
    if last.resolve_ts - last.ts < 900: prov[(ed, k)] = int(last.spot > k)
b["outcome"] = [out.get((ed, k), prov.get((ed, k))) for ed, k in zip(b.event_date, b.strike)]
b["src"] = ["resolved" if (ed, k) in out else ("provisional" if (ed, k) in prov else None) for ed, k in zip(b.event_date, b.strike)]
all_days = sorted(b.event_date.unique())

q = b[b.pct.between(*a.pct) & b.hours.between(*a.hours)].copy()
print(f"{a.asset} above-family snapshots {len(b):,} on {len(all_days)} event-days; rule {a.pct[0]:.0f}-{a.pct[1]:.0f}% above strike, "
      f"{a.hours[0]:.0f}-{a.hours[1]:.0f}h out, Yes <= {a.max_q}: {len(q):,} qualifying snapshots, {q.groupby(['event_date','strike']).ngroups} strike-days")
if q.empty: raise SystemExit("no qualifying snapshots")

# ---- QUEUE
q["queue"] = np.select([q.best_bid.isna(), q.best_bid < a.max_q - EPS, (q.best_bid - a.max_q).abs() < EPS],
                       ["empty book", "we'd be best bid", "join"], default="behind a higher bid")
q["ok"] = q.best_ask <= a.max_q + EPS
print(f"\nQUEUE for a resting Yes bid at {a.max_q}:"); print(q.queue.value_counts().to_string())
print(f"Yes ask <= {a.max_q} in {q.ok.mean():.1%} of snapshots; median ask {q.best_ask.median():.4f}, median bid {q.best_bid.median():.4f}, "
      f"median spread {(q.best_ask - q.best_bid).median():.4f}")

# ---- TAKER (book-walked)
q["lev"] = [[(p, p * z) for p, z in sorted(json.loads(j)) if 0 < p <= a.max_q + EPS] for j in q.asks_json]
rows = q[q.ok].sort_values("ts").groupby(["event_date", "strike"]).head(1)
taken, spent = [], {}
for r in rows.itertuples():
    rem = a.cap - spent.get(r.event_date, 0.0); cost = shares = 0.0
    for px, usd in r.lev:
        if rem <= EPS: break
        take = min(usd, rem); cost += take; shares += take / px; rem -= take
    if cost <= 0: continue
    spent[r.event_date] = spent.get(r.event_date, 0.0) + cost
    pnl = None if pd.isna(r.outcome) else shares * r.outcome - cost
    taken.append((r.event_date, r.strike, datetime.fromtimestamp(r.ts, NY).strftime("%m-%d %H:%M"), round(r.pct, 2), round(r.hours, 1),
                  round(cost / shares, 4), round(cost), r.outcome, r.src, None if pnl is None else round(pnl, 2)))
t = pd.DataFrame(taken, columns=["event", "strike", "entry(NY)", "pct", "hrs", "vwap", "usd", "outcome", "src", "pnl"])
print("\n--- TAKER paper (book-walked VWAP, first qualifying snapshot per strike-day) ---")
if t.empty:
    print("no entries")
else:
    print(t.to_string(index=False))
    d = t.groupby("event").agg(usd=("usd", "sum"), pnl=("pnl", "sum"))
    print(f"days {len(d)}/{len(all_days)}  avg deployed ${d.usd.mean():,.0f}/day  P&L ${d.pnl.sum():+,.0f} on ${d.usd.sum():,.0f} = {d.pnl.sum()/max(d.usd.sum(),1):+.2%}"
          f"  losses {int((t.pnl < 0).sum())}/{len(t)}  (provisional: {int((t.src == 'provisional').sum())}, unresolved: {int(t.outcome.isna().sum())})")

# ---- MAKER
if a.fills:
    f = dd.execute("""SELECT m.event_date, m.strike, f.ts, f.price, f.size, f.side, (f.token = m.yes_token) AS is_yes
                      FROM fills f JOIN markets m USING (condition_id)""").df()
    f["event_date"] = f.event_date.astype(str).str[:10]
    f["p_yes"] = np.where(f.is_yes, f.price, 1 - f.price)
    f["taker_buys_yes"] = np.where(f.is_yes, f.side == "BUY", f.side == "SELL")
    win = q.groupby(["event_date", "strike"]).agg(t0=("ts", "min"), t1=("ts", "max")).reset_index()
    omap = b.dropna(subset=["outcome"]).groupby(["event_date", "strike"]).outcome.first().reset_index()
    g = f.merge(win, on=["event_date", "strike"])
    g = g[(g.ts >= g.t0) & (g.ts <= g.t1 + 120) & ~g.taker_buys_yes & (g.p_yes <= a.max_q + EPS)].merge(omap, on=["event_date", "strike"], how="left")
    print(f"\n--- MAKER: taker sells of Yes at <= {a.max_q} inside the cell's windows (ceiling for a maker) ---")
    if g.empty:
        print("no qualifying prints (re-pull fills if the logged days are newer than the fills DB)")
    else:
        g["usd"] = g["size"] * g.p_yes; g["pnl"] = g["size"] * (g.outcome - g.p_yes)
        g["px_b"] = pd.cut(g.p_yes, [0, 0.90, 0.95, a.max_q + EPS])
        print(g.groupby("px_b", observed=True).agg(prints=("usd", "size"), usd=("usd", "sum"), vw_px=("p_yes", lambda x: np.average(x, weights=g.loc[x.index, "usd"])),
                                                   lost_share=("outcome", lambda x: 1 - x.mean()), pnl=("pnl", "sum")).round(4).to_string())
        d = g.groupby("event_date").agg(usd=("usd", "sum"), pnl=("pnl", "sum")).reindex(all_days, fill_value=0)
        sd = g.groupby(["event_date", "strike"]).outcome.first()
        print(f"total ${g.usd.sum():,.0f} over {len(all_days)} logged days = ${g.usd.sum()/len(all_days):,.0f}/day (median day ${d.usd.median():,.0f}); "
              f"P&L as maker on every print ${g.pnl.sum():+,.0f} = {g.pnl.sum()/max(g.usd.sum(),1):+.2%}; worst day ${d.pnl.min():+,.0f}; "
              f"strike-days lost {int((sd == 0).sum())}/{len(sd)}")
