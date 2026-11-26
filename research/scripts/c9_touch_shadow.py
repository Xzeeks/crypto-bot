#!/usr/bin/env python3
"""c9_touch_shadow.py — score the touch-ladder shapes against logged order books (logger/book_logger.py, family='hit').

    python research/scripts/c9_touch_shadow.py research/data/books.sqlite [--asset ethereum] [--fills research/data/eth_hit.duckdb]

Books are logged on the Yes token of each barrier, only while the barrier is alive (Polymarket pulls the book the moment
a barrier is touched; the logger records that as resolve_ts < window end, with last_seen = hit time).

SELL shapes (buy NO = bet the barrier is NOT touched before 23:59 ET)
    ↑ barriers 3-7% above spot, 3-9h left;  ↓ barriers 4-7% below spot, 2-6h left;  NO price <= --sell-max-q (0.975)
BUY shape   (buy Yes = bet the barrier IS touched): either side, 3-5% away, 12-18h left, Yes price <= --buy-max-p (0.13)

Per shape:
  QUEUE  where a resting order at the limit price would sit in the logged book.
  TAKER  paper: at the FIRST qualifying snapshot of each barrier-day, walk the book levels inside the limit (true VWAP),
         up to --cap $ per event-day.
  MAKER  (needs --fills): taker prints that hit resting orders on the cell's barriers INSIDE the cell's own time window
         (first to last qualifying snapshot of that barrier-day), with the P&L of having been the maker on every print at
         its price. This is the ceiling for a maker; QUEUE says how often we would have been first in line.
Outcome: touched iff the logger's early-resolve mark (markets.resolve_ts < window end) or the fills DB says Yes.
"""
import argparse, json, sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
NY = ZoneInfo("America/New_York")

ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("--asset", default="ethereum")
ap.add_argument("--fills", default=None); ap.add_argument("--cap", type=float, default=5000)
ap.add_argument("--sell-max-q", type=float, default=0.975); ap.add_argument("--buy-max-p", type=float, default=0.13)
a = ap.parse_args()
EPS = 1e-9

con = sqlite3.connect(a.db)
b = pd.read_sql("SELECT * FROM books WHERE family='hit' AND asset=? AND spot IS NOT NULL", con, params=(a.asset,))
if b.empty: raise SystemExit("no hit-family books logged yet for " + a.asset)
m = pd.read_sql("SELECT event_date, direction, strike, resolve_ts, last_seen FROM markets WHERE family='hit' AND asset=?", con, params=(a.asset,))
def wend(ed):
    d = datetime.fromisoformat(ed); return int((datetime(d.year, d.month, d.day, tzinfo=NY) + timedelta(days=1)).timestamp())
m["window_end"] = m.event_date.map(wend)
# The logger marks a pulled book (CLOB 404) by setting resolve_ts = time_of_404 - 7200. Untouched barriers are pulled too,
# a few minutes AFTER the window ends, so "pulled" alone does not mean touched: touched = pulled BEFORE the window end.
m["pulled"] = m.resolve_ts != m.window_end
m["touched_log"] = m.pulled & (m.resolve_ts + 7200 < m.window_end)
m["closed_log"] = m.pulled | (m.last_seen >= m.window_end - 600)
b["window_end"] = b.event_date.map(wend)
b["hours"] = (b.window_end - b.ts) / 3600
b["pct"] = np.where(b.direction == "↑", (b.strike - b.spot) / b.spot * 100, (b.spot - b.strike) / b.spot * 100)
b = b[b.pct > 0]
b = b.merge(m[["event_date", "direction", "strike", "touched_log", "closed_log"]], on=["event_date", "direction", "strike"], how="left")
out = {}
if a.fills:
    import duckdb
    dd = duckdb.connect(a.fills, read_only=True)
    for ed, d_, k, o in dd.execute("SELECT event_date, direction, strike, outcome FROM markets WHERE outcome IS NOT NULL").fetchall():
        out[(str(ed)[:10], d_, float(k))] = int(o)
b["touched"] = [out.get((ed, d_, k), (int(t) if c else None)) for ed, d_, k, t, c in zip(b.event_date, b.direction, b.strike, b.touched_log, b.closed_log)]
all_days = sorted(b.event_date.unique())
print(f"{a.asset} hit-family snapshots {len(b):,} on {len(all_days)} event-days; outcomes known for "
      f"{b.dropna(subset=['touched']).groupby(['event_date','direction','strike']).ngroups} barriers")


def paper(rows, lev_col, win_fn, label):
    """First qualifying snapshot per barrier-day; walk levels [(px, usd), ...] inside the limit; daily cap."""
    rows = rows.sort_values("ts").groupby(["event_date", "direction", "strike"]).head(1)
    taken, spent = [], {}
    for r in rows.itertuples():
        rem = a.cap - spent.get(r.event_date, 0.0)
        cost = shares = 0.0
        for px, usd in getattr(r, lev_col):
            if rem <= EPS: break
            take = min(usd, rem); cost += take; shares += take / px; rem -= take
        if cost <= 0: continue
        spent[r.event_date] = spent.get(r.event_date, 0.0) + cost
        vwap = cost / shares
        pnl = None if pd.isna(r.touched) else shares * win_fn(r.touched) - cost
        taken.append((r.event_date, r.direction, r.strike, datetime.fromtimestamp(r.ts, NY).strftime("%m-%d %H:%M"), round(r.pct, 2), round(r.hours, 1),
                      round(vwap, 4), round(cost), r.touched, None if pnl is None else round(pnl, 2)))
    t = pd.DataFrame(taken, columns=["event", "dir", "strike", "entry(NY)", "pct", "hrs", "vwap", "usd", "touched", "pnl"])
    print(f"\n--- {label} — TAKER paper (book-walked VWAP) ---")
    if t.empty: print("no entries"); return
    print(t.to_string(index=False))
    d = t.groupby("event").agg(usd=("usd", "sum"), pnl=("pnl", "sum"))
    print(f"days {len(d)}/{len(all_days)}  avg deployed ${d.usd.mean():,.0f}/day  P&L ${d.pnl.sum():+,.0f} on ${d.usd.sum():,.0f} = {d.pnl.sum()/max(d.usd.sum(),1):+.2%}"
          f"  losses {int((t.pnl < 0).sum())}/{len(t)}  (unresolved: {int(t.touched.isna().sum())})")


cells = []   # (label, kind, qualifying snapshots)

# ================= SELL shapes =================
yes_ask_we_rest = round(1 - a.sell_max_q, 4)
for label, side, plo, phi, hlo, hhi in [("SELL ↑ (reach) 3-7% above, 3-9h", "↑", 3, 7, 3, 9), ("SELL ↓ (dip) 4-7% below, 2-6h", "↓", 4, 7, 2, 6)]:
    s = b[(b.direction == side) & b.pct.between(plo, phi) & b.hours.between(hlo, hhi)].copy()
    print(f"\n=== {label}: {len(s):,} snapshots, {s.groupby(['event_date','strike']).ngroups} barrier-days ===")
    if s.empty: continue
    cells.append((label, "sell", s))
    s["no_px_taker"] = 1 - s.best_bid
    s["ok"] = s.no_px_taker <= a.sell_max_q + EPS
    s["queue"] = np.select([s.best_ask.isna(), s.best_ask > yes_ask_we_rest + EPS, (s.best_ask - yes_ask_we_rest).abs() < EPS],
                           ["empty book", "we'd be best ask", "join"], default="behind a lower ask")
    print("QUEUE for a resting Yes ask at", yes_ask_we_rest, ":"); print(s.queue.value_counts().to_string())
    print(f"taker NO price <= {a.sell_max_q} available in {s.ok.mean():.1%} of snapshots; median NO taker px {s.no_px_taker.median():.4f}; "
          f"median Yes spread {(s.best_ask - s.best_bid).median():.4f}")
    # selling Yes into bids = buying NO at 1-p; best (highest) bid first
    s["lev"] = [[(1 - p, (1 - p) * z) for p, z in sorted(json.loads(j), reverse=True) if 1 - p <= a.sell_max_q + EPS and p < 1] for j in s.bids_json]
    paper(s[s.ok], "lev", lambda touched: 0.0 if touched else 1.0, label)

# ================= BUY shape =================
u = b[b.pct.between(3, 5) & b.hours.between(12, 18)].copy()
label = "BUY touch, either side, 3-5% away, 12-18h"
print(f"\n=== {label}: {len(u):,} snapshots, {u.groupby(['event_date','direction','strike']).ngroups} barrier-days ===")
if not u.empty:
    cells.append((label, "buy", u))
    u["ok"] = u.best_ask <= a.buy_max_p + EPS
    u["queue"] = np.select([u.best_bid.isna(), u.best_bid < a.buy_max_p - EPS, (u.best_bid - a.buy_max_p).abs() < EPS],
                           ["empty book", "we'd be best bid", "join"], default="behind a higher bid")
    print("QUEUE for a resting Yes bid at", a.buy_max_p, ":"); print(u.queue.value_counts().to_string())
    print(f"Yes ask <= {a.buy_max_p} in {u.ok.mean():.1%} of snapshots; median Yes ask {u.best_ask.median():.4f}, median bid {u.best_bid.median():.4f}")
    u["lev"] = [[(p, p * z) for p, z in sorted(json.loads(j)) if 0 < p <= a.buy_max_p + EPS] for j in u.asks_json]
    paper(u[u.ok], "lev", lambda touched: 1.0 if touched else 0.0, label)

# ================= MAKER: prints inside each cell's own windows =================
if a.fills:
    f = dd.execute("""SELECT m.event_date, m.direction, m.strike, f.ts, f.price, f.size, f.side, (f.token = m.yes_token) AS is_yes
                      FROM fills f JOIN markets m USING (condition_id)""").df()
    f["event_date"] = f.event_date.astype(str).str[:10]
    f["p_yes"] = np.where(f.is_yes, f.price, 1 - f.price)
    f["taker_buys_yes"] = np.where(f.is_yes, f.side == "BUY", f.side == "SELL")
    tmap = b.dropna(subset=["touched"]).groupby(["event_date", "direction", "strike"]).touched.first().reset_index()
    print("\n=== MAKER: taker prints that hit resting orders inside each cell's window (ceiling for a maker) ===")
    for label, kind, s in cells:
        win = s.groupby(["event_date", "direction", "strike"]).agg(t0=("ts", "min"), t1=("ts", "max")).reset_index()
        g = f.merge(win, on=["event_date", "direction", "strike"])
        g = g[(g.ts >= g.t0) & (g.ts <= g.t1 + 120)].merge(tmap, on=["event_date", "direction", "strike"], how="left")
        if kind == "sell":   # takers BUY Yes from resting asks -> the maker is long NO at 1 - p_yes
            g = g[g.taker_buys_yes & (g.p_yes >= yes_ask_we_rest - EPS)].copy()
            g["px"] = 1 - g.p_yes; g["usd"] = g["size"] * g.px
            g["pnl"] = g["size"] * ((1 - g.touched) - g.px)
            bins = [0, 0.90, 0.95, 0.975 + EPS]
        else:                # takers SELL Yes into resting bids -> the maker is long Yes at p_yes
            g = g[~g.taker_buys_yes & (g.p_yes <= a.buy_max_p + EPS)].copy()
            g["px"] = g.p_yes; g["usd"] = g["size"] * g.px
            g["pnl"] = g["size"] * (g.touched - g.px)
            bins = [0, 0.05, 0.10, a.buy_max_p + EPS]
        print(f"\n--- {label} ---")
        if g.empty: print("no qualifying prints"); continue
        g["px_b"] = pd.cut(g.px, bins)
        print(g.groupby("px_b", observed=True).agg(prints=("usd", "size"), usd=("usd", "sum"), vw_px=("px", lambda x: np.average(x, weights=g.loc[x.index, "usd"])),
                                                   touched_share=("touched", "mean"), pnl=("pnl", "sum")).round(4).to_string())
        d = g.groupby("event_date").agg(usd=("usd", "sum"), pnl=("pnl", "sum")).reindex(all_days, fill_value=0)
        print(f"total ${g.usd.sum():,.0f} over {len(all_days)} logged days = ${g.usd.sum()/len(all_days):,.0f}/day (median day ${d.usd.median():,.0f}); "
              f"P&L as maker on every print ${g.pnl.sum():+,.0f} = {g.pnl.sum()/max(g.usd.sum(),1):+.2%}; worst day ${d.pnl.min():+,.0f}; "
              f"barrier-days touched {int(g.groupby(['event_date','direction','strike']).touched.first().sum())}/{g.groupby(['event_date','direction','strike']).ngroups}")
