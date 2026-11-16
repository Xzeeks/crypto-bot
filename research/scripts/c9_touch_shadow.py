#!/usr/bin/env python3
"""c9_touch_shadow.py — score the two touch-ladder shapes against logged order books (logger/book_logger.py, family='hit').

    python research/scripts/c9_touch_shadow.py research/data/books.sqlite [--asset ethereum] [--fills research/data/eth_hit.duckdb]

Books are logged on the Yes token of each barrier, only while the barrier is alive (Polymarket pulls the book the moment
a barrier is touched; the logger records that as resolve_ts < window end, with last_seen = hit time).

SELL shape  (buy NO = bet the barrier is NOT touched before 23:59 ET)
    ↑ barriers 3-7% above spot, 3-9h left, NO price <= --sell-max-q (default 0.975)   [also ↓ 4-7% below, 2-6h, as a 2nd line]
    taker: hit the Yes bid (sell Yes) -> NO price = 1 - best_bid.   maker: rest a Yes ask at 1 - sell_max_q.
BUY shape   (buy Yes = bet the barrier IS touched)
    either side, 3-5% away, 12-18h left, Yes ask <= --buy-max-p (default 0.13)
    taker: lift the Yes ask.   maker: rest a Yes bid at buy_max_p.
Outcome: touched iff the barrier got the early-resolve mark (markets.resolve_ts < window end) or the fills DB says Yes.
Reports per shape: QUEUE (would our resting order be first?), TAKER paper P&L with a daily cap, MAKER capture from the fills DB.
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

con = sqlite3.connect(a.db)
b = pd.read_sql("SELECT * FROM books WHERE family='hit' AND asset=? AND spot IS NOT NULL", con, params=(a.asset,))
if b.empty: raise SystemExit("no hit-family books logged yet for " + a.asset)
m = pd.read_sql("SELECT event_date, direction, strike, resolve_ts, last_seen FROM markets WHERE family='hit' AND asset=?", con, params=(a.asset,))
def wend(ed):
    d = datetime.fromisoformat(ed); return int((datetime(d.year, d.month, d.day, tzinfo=NY) + timedelta(days=1)).timestamp())
m["window_end"] = m.event_date.map(wend)
m["touched_log"] = m.resolve_ts < m.window_end - 3600          # early-resolve mark
m["closed_log"] = (m.last_seen >= m.window_end - 600) | m.touched_log   # we observed the end (or the hit)
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
        out[(str(ed), d_, float(k))] = int(o)
b["touched"] = [out.get((ed, d_, k), (int(t) if c else None)) for ed, d_, k, t, c in zip(b.event_date, b.direction, b.strike, b.touched_log, b.closed_log)]
print(f"{a.asset} hit-family snapshots {len(b):,} on {b.event_date.nunique()} event-days; outcomes known for "
      f"{b.dropna(subset=['touched']).groupby(['event_date','direction','strike']).ngroups} barriers")
all_days = sorted(b.event_date.unique())

def paper(rows, px_col, usd_col, pnl_fn, label):
    rows = rows.sort_values("ts").groupby(["event_date", "direction", "strike"]).head(1)
    taken, spent = [], {}
    for r in rows.itertuples():
        s = spent.get(r.event_date, 0.0)
        if s >= a.cap: continue
        usd = min(getattr(r, usd_col), a.cap - s); spent[r.event_date] = s + usd
        taken.append((r.event_date, r.direction, r.strike, datetime.fromtimestamp(r.ts, NY).strftime("%m-%d %H:%M"), round(r.pct, 2), round(r.hours, 1),
                      getattr(r, px_col), round(usd), r.touched, None if r.touched is None else round(pnl_fn(usd, getattr(r, px_col), r.touched), 2)))
    t = pd.DataFrame(taken, columns=["event", "dir", "strike", "entry(NY)", "pct", "hrs", "px", "usd", "touched", "pnl"])
    print(f"\n--- {label} ---")
    if t.empty: print("no entries"); return
    print(t.to_string(index=False))
    d = t.groupby("event").agg(usd=("usd", "sum"), pnl=("pnl", "sum"))
    print(f"days {len(d)}/{len(all_days)}  avg deployed ${d.usd.mean():,.0f}/day  P&L ${d.pnl.sum():+,.0f} on ${d.usd.sum():,.0f} = {d.pnl.sum()/max(d.usd.sum(),1):+.2%}"
          f"  (unresolved: {int(t.touched.isna().sum())})")

# ================= SELL shape =================
for label, side, plo, phi, hlo, hhi in [("SELL ↑ (reach) 3-7% above, 3-9h", "↑", 3, 7, 3, 9), ("SELL ↓ (dip) 4-7% below, 2-6h", "↓", 4, 7, 2, 6)]:
    s = b[(b.direction == side) & b.pct.between(plo, phi) & b.hours.between(hlo, hhi)].copy()
    print(f"\n=== {label}: {len(s):,} snapshots, {s.groupby(['event_date','strike']).ngroups} barrier-days ===")
    if s.empty: continue
    s["no_px_taker"] = 1 - s.best_bid                       # sell Yes into the bid = buy NO at 1 - bid
    s["ok"] = s.no_px_taker <= a.sell_max_q
    yes_ask_we_rest = round(1 - a.sell_max_q, 4)                      # maker: our Yes ask
    s["queue"] = np.select([s.best_ask.isna(), s.best_ask > yes_ask_we_rest + 1e-9, (s.best_ask - yes_ask_we_rest).abs() < 1e-9],
                           ["empty book", "we'd be best ask", "join"], default="behind a lower ask")
    print("QUEUE for a resting Yes ask at", yes_ask_we_rest, ":"); print(s.queue.value_counts().to_string())
    print(f"taker NO price <= {a.sell_max_q} available in {s.ok.mean():.1%} of snapshots; median NO taker px {s.no_px_taker.median():.4f}; "
          f"median Yes spread {(s.best_ask - s.best_bid).median():.4f}")
    bids = s.bids_json.map(json.loads)
    s["usd_bid"] = [sum((1 - p) * z for p, z in lv if 1 - p <= a.sell_max_q + 1e-9) for lv in bids]   # NO notional we could buy by hitting bids
    paper(s[s.ok], "no_px_taker", "usd_bid", lambda usd, px, touched: usd / px * ((0 if touched else 1) - px), label + " — TAKER paper")

# ================= BUY shape =================
u = b[b.pct.between(3, 5) & b.hours.between(12, 18)].copy()
print(f"\n=== BUY touch, either side, 3-5% away, 12-18h: {len(u):,} snapshots, {u.groupby(['event_date','direction','strike']).ngroups} barrier-days ===")
if not u.empty:
    u["ok"] = u.best_ask <= a.buy_max_p
    u["queue"] = np.select([u.best_bid.isna(), u.best_bid < a.buy_max_p - 1e-9, (u.best_bid - a.buy_max_p).abs() < 1e-9],
                           ["empty book", "we'd be best bid", "join"], default="behind a higher bid")
    print("QUEUE for a resting Yes bid at", a.buy_max_p, ":"); print(u.queue.value_counts().to_string())
    print(f"Yes ask <= {a.buy_max_p} in {u.ok.mean():.1%} of snapshots; median Yes ask {u.best_ask.median():.4f}, median bid {u.best_bid.median():.4f}")
    asks = u.asks_json.map(json.loads)
    u["usd_ask"] = [sum(p * z for p, z in lv if p <= a.buy_max_p + 1e-9) for lv in asks]
    paper(u[u.ok], "best_ask", "usd_ask", lambda usd, px, touched: usd / px * ((1 if touched else 0) - px), "BUY touch — TAKER paper")

# ================= MAKER capture from fills =================
if a.fills:
    f = dd.execute("""SELECT m.event_date, m.direction, m.strike, f.ts, f.price, f.size, f.side, (f.token = m.yes_token) AS is_yes
                      FROM fills f JOIN markets m USING (condition_id)""").df()
    f["event_date"] = f.event_date.astype(str); f["p_yes"] = np.where(f.is_yes, f.price, 1 - f.price)
    f["taker_buys_yes"] = np.where(f.is_yes, f.side == "BUY", f.side == "SELL")
    win = b.groupby(["event_date", "direction", "strike"]).agg(t0=("ts", "min"), t1=("ts", "max")).reset_index()
    f = f.merge(win, on=["event_date", "direction", "strike"]); f = f[(f.ts >= f.t0) & (f.ts <= f.t1)]
    sell_flow = f[f.taker_buys_yes & (f.p_yes >= 1 - a.sell_max_q - 1e-9)]      # takers buying Yes at >= our ask: fills our SELL-shape maker order
    buy_flow = f[~f.taker_buys_yes & (f.p_yes <= a.buy_max_p + 1e-9)]           # takers selling Yes at <= our bid: fills our BUY-shape maker order
    print(f"\n=== MAKER capture (taker prints on logged barriers while we'd have been live) ===")
    print(f"SELL shape: ${(sell_flow['size'] * (1 - sell_flow.p_yes)).sum():,.0f} of NO notional hit asks at Yes >= {1 - a.sell_max_q:.3f} "
          f"over {len(all_days)} logged days")
    print(f"BUY shape:  ${(buy_flow['size'] * buy_flow.p_yes).sum():,.0f} of Yes sold into bids at <= {a.buy_max_p} over {len(all_days)} logged days")
