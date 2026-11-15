#!/usr/bin/env python3
"""c6_shadow_eval.py — evaluate the ETH downside-carry rule against logged order books (from logger/book_logger.py).

    python research/scripts/c6_shadow_eval.py research/data/books.sqlite [--asset ethereum] [--pct 5 8] [--hours 12 24]
                                              [--max-q 0.975] [--cap 5000] [--fills research/data/eth_above.duckdb]

Rule (per snapshot): asset spot is 5-8% ABOVE a strike, 12-24h before the noon print -> we want Yes on that strike.
Three execution models, all from the same snapshots:
  TAKER   lift the best Yes ask when ask <= max_q. Fills for sure; pays the spread.
  MAKER   rest a Yes bid at max_q. Counts as "fillable" only if, in the fills DB (optional), a taker SELL of Yes (or
          BUY of No) printed at <= max_q on that strike after our bid would have been live. Needs --fills.
  QUEUE   how often our max_q bid would be best bid (best_bid < max_q) vs joining/behind an existing bid at >= max_q:
          that is the capture-rate question — if a 0.975 bid is already standing there, we are not first in line.
Outcomes come from the fills DB markets table when the event has resolved; otherwise from spot at the logged print
(last snapshot before resolve_ts) as a provisional read, flagged.
Reports: per-day opportunity ($ available at <= max_q on the ask, best-bid state), paper P&L per model with the daily
cap, and the book shape at the moment of each qualifying snapshot.
"""
import argparse, json, sqlite3, math
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
NY = ZoneInfo("America/New_York")

ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("--asset", default="ethereum")
ap.add_argument("--pct", nargs=2, type=float, default=[5, 8]); ap.add_argument("--hours", nargs=2, type=float, default=[12, 24])
ap.add_argument("--max-q", type=float, default=0.975); ap.add_argument("--cap", type=float, default=5000)
ap.add_argument("--fills", default=None, help="duckdb from c1 pull, for resolved outcomes + taker prints (maker model)")
a = ap.parse_args()

con = sqlite3.connect(a.db)
b = pd.read_sql("SELECT * FROM books WHERE family='above' AND asset=? AND spot IS NOT NULL", con, params=(a.asset,))
if b.empty: raise SystemExit("no books logged yet for " + a.asset)
b["hours"] = (b.resolve_ts - b.ts) / 3600
b["pct"] = (b.spot - b.strike) / b.strike * 100           # positive = spot above strike = we want Yes
q = b[(b.pct.between(*a.pct)) & (b.hours.between(*a.hours))].copy()
print(f"snapshots: {len(b):,} over {b.ts.nunique():,} ticks, {b.event_date.nunique()} events; qualifying snapshots: {len(q):,} on {q.event_date.nunique()} events")

# outcomes
out = {}
if a.fills:
    import duckdb
    dd = duckdb.connect(a.fills, read_only=True)
    for ed, k, o in dd.execute("SELECT event_date, strike, outcome FROM markets WHERE outcome IS NOT NULL").fetchall():
        out[(str(ed), float(k))] = int(o)
prov = {}
for (ed, k), g in b.groupby(["event_date", "strike"]):
    last = g[g.ts <= g.resolve_ts.iloc[0]].sort_values("ts").tail(1)
    if len(last) and (g.resolve_ts.iloc[0] - last.ts.iloc[0]) < 900:
        prov[(ed, k)] = int(last.spot.iloc[0] > k)
q["outcome"] = [out.get((ed, k), prov.get((ed, k))) for ed, k in zip(q.event_date, q.strike)]
q["outcome_src"] = ["resolved" if (ed, k) in out else ("provisional" if (ed, k) in prov else None) for ed, k in zip(q.event_date, q.strike)]

# ---- QUEUE: where would a bid at max_q sit?
q["bid_state"] = np.select([q.best_bid.isna(), q.best_bid < a.max_q - 1e-9, q.best_bid.sub(a.max_q).abs() < 1e-9],
                           ["empty book", "we'd be best bid", "join at max_q"], default="behind a higher bid")
q["ask_le_maxq"] = q.best_ask <= a.max_q + 1e-9
print("\n=== QUEUE: state of the Yes book at qualifying snapshots ===")
print(q.bid_state.value_counts().to_string())
print(f"\nbest ask <= {a.max_q}: {q.ask_le_maxq.mean():.1%} of qualifying snapshots; "
      f"median best ask {q.best_ask.median():.4f}, median best bid {q.best_bid.median():.4f}, median spread {(q.best_ask-q.best_bid).median():.4f}")
asks = q.asks_json.map(json.loads)
q["ask_usd_le_maxq"] = [sum(p * s for p, s in lv if p <= a.max_q + 1e-9) for lv in asks]
bids = q.bids_json.map(json.loads)
q["bid_usd_ge_maxq"] = [sum(p * s for p, s in lv if p >= a.max_q - 1e-9) for lv in bids]
print(f"$ resting on the ask at <= {a.max_q} when present: median ${q.loc[q.ask_le_maxq,'ask_usd_le_maxq'].median():,.0f}, "
      f"p90 ${q.loc[q.ask_le_maxq,'ask_usd_le_maxq'].quantile(.9):,.0f}")
print(f"$ already bid at >= {a.max_q} ahead of us when 'behind/join': median ${q.loc[q.bid_state!='we\'d be best bid','bid_usd_ge_maxq'].median():,.0f}")

# ---- TAKER paper: first qualifying snapshot per (event, strike) with ask <= max_q; size = min(ask $ at <= max_q, cap remaining)
print("\n=== TAKER paper (lift the ask at <= max_q, once per strike per day, $cap/day) ===")
t = q[q.ask_le_maxq].sort_values("ts").groupby(["event_date", "strike"]).head(1)
rows, spent = [], {}
for r in t.itertuples():
    s = spent.get(r.event_date, 0.0)
    if s >= a.cap: continue
    usd = min(r.ask_usd_le_maxq, a.cap - s); spent[r.event_date] = s + usd
    px = r.best_ask  # conservative: whole size at best ask (real VWAP across levels <= max_q would be slightly worse)
    pnl = None if r.outcome is None else usd / px * (r.outcome - px)
    rows.append((r.event_date, r.strike, datetime.fromtimestamp(r.ts, NY).strftime("%m-%d %H:%M"), round(r.pct, 2), round(r.hours, 1), px, round(usd), r.outcome, r.outcome_src, None if pnl is None else round(pnl, 2)))
tp = pd.DataFrame(rows, columns=["event", "strike", "entry(NY)", "pct", "hrs", "px", "usd", "outcome", "src", "pnl"])
print(tp.to_string(index=False) if len(tp) else "no taker entries")
if len(tp):
    d = tp.groupby("event").agg(usd=("usd", "sum"), pnl=("pnl", "sum"))
    print(f"\ndays with entries {len(d)}/{q.event_date.nunique()}, avg deployed ${d.usd.mean():,.0f}/day, "
          f"P&L ${d.pnl.sum():+,.0f} on ${d.usd.sum():,.0f} = {d.pnl.sum()/max(d.usd.sum(),1):+.2%}  (provisional outcomes: {int((tp.src=='provisional').sum())})")

# ---- MAKER capture: did taker prints at <= max_q happen on our strikes while we'd have been live?
if a.fills:
    print(f"\n=== MAKER capture (taker sells of Yes at <= {a.max_q} on qualifying strikes during the window) ===")
    f = dd.execute("""SELECT m.event_date, m.strike, f.ts, f.price, f.size, f.side, (f.token = m.yes_token) AS is_yes
                      FROM fills f JOIN markets m USING (condition_id)""").df()
    f["event_date"] = f.event_date.astype(str)
    f["p_yes"] = np.where(f.is_yes, f.price, 1 - f.price)
    f["taker_sells_yes"] = np.where(f.is_yes, f.side == "SELL", f.side == "BUY")
    win = q.groupby(["event_date", "strike"]).agg(t0=("ts", "min"), t1=("ts", "max")).reset_index()
    f = f.merge(win, on=["event_date", "strike"])
    f = f[(f.ts >= f.t0) & (f.ts <= f.t1) & f.taker_sells_yes & (f.p_yes <= a.max_q + 1e-9)]
    f["usd"] = f["size"] * f.p_yes
    if f.empty:
        print("no taker sells at <= max_q printed on qualifying strikes during logged windows (fills DB may lag — re-pull)")
    else:
        m = f.groupby("event_date").agg(prints=("usd", "size"), usd=("usd", "sum"), avg_px=("p_yes", "mean"))
        print(m.round(3).to_string()); print(f"\nmaker-side flow on qualifying strikes: ${m.usd.sum():,.0f} over {len(m)} days = ${m.usd.sum()/max(q.event_date.nunique(),1):,.0f}/logged day")
        print("Interpretation: this is the $ a resting bid at <= max_q could have absorbed; compare with QUEUE above for whether ours would have been first.")

# ---- book shape
print("\n=== book shape at qualifying snapshots (Yes side), by hours-out ===")
q["h_b"] = pd.cut(q.hours, [12, 15, 18, 21, 24])
print(q.groupby("h_b", observed=True).agg(n=("ts", "size"), best_bid=("best_bid", "median"), best_ask=("best_ask", "median"),
      ask_le=("ask_le_maxq", "mean"), ask_usd=("ask_usd_le_maxq", "median"), spot_pct=("pct", "median")).round(4).to_string())
