#!/usr/bin/env python3
"""c11_shadow_review.py — one collective review of every shadow trade: all cells, both assets, day by day.

    python research/scripts/c11_shadow_review.py research/data/books.sqlite [--data-dir research/data] [--cap 5000]

Expects the fills DBs in --data-dir: eth_above.duckdb, btc_above.duckdb, eth_hit.duckdb, btc_hit.duckdb (outcomes + taker prints).

Cells (the shadow set):
  carry    close ladder: buy Yes on strikes 5-8% (ETH) / 4-8% (BTC) below spot, 12-24h before noon ET, Yes <= 0.975
  sellDn   touch ladder: buy NO on ↓ barriers 4-7% below spot, 2-6h left, NO <= 0.975
  sellUp   touch ladder: buy NO on ↑ barriers 3-7% above spot, 3-9h left, NO <= 0.975
  buyT     touch ladder: buy Yes on barriers 3-5% away (either side), 12-18h left, Yes <= 0.13
TAKER paper = what could have been executed against the logged book: first qualifying snapshot per market-day, walking the
book levels inside the limit, up to --cap $ per cell per event-day. MAKER = ceiling: every taker print that hit a resting order
inside the cell's window, with the maker's P&L on it.
Days are event dates (the day the market resolves). Unresolved trades are listed as pending and excluded from P&L.
"""
import argparse, json, os, sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import duckdb, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
NY = ZoneInfo("America/New_York"); EPS = 1e-9
ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("--data-dir", default="research/data"); ap.add_argument("--cap", type=float, default=5000)
ap.add_argument("--trades", action="store_true", help="also print every individual trade")
a = ap.parse_args()

CELLS = [
    dict(code="carry",  fam="above", side="yes", limit=0.975, hours=(12, 24), pct={"ethereum": (5, 8), "bitcoin": (4, 8)}, dirs=None),
    dict(code="sellDn", fam="hit",   side="no",  limit=0.975, hours=(2, 6),   pct=(4, 7), dirs="↓"),
    dict(code="sellUp", fam="hit",   side="no",  limit=0.975, hours=(3, 9),   pct=(3, 7), dirs="↑"),
    dict(code="buyT",   fam="hit",   side="yes", limit=0.13,  hours=(12, 18), pct=(3, 5), dirs=None),
]
SHORT = {"ethereum": "ETH", "bitcoin": "BTC"}; PRE = {"ethereum": "eth", "bitcoin": "btc"}
KEYS = ["event_date", "direction", "strike"]
def wend(ed):
    d = datetime.fromisoformat(ed); return int((datetime(d.year, d.month, d.day, tzinfo=NY) + timedelta(days=1)).timestamp())

con = sqlite3.connect(a.db)
trades, makers, days_logged = [], [], set()
for asset in ("ethereum", "bitcoin"):
    for fam in ("above", "hit"):
        path = os.path.join(a.data_dir, f"{PRE[asset]}_{fam}.duckdb")
        if not os.path.exists(path): print(f"missing {path} — skipping {asset} {fam}"); continue
        b = pd.read_sql("SELECT ts, event_date, direction, strike, resolve_ts, spot, best_bid, best_ask, bids_json, asks_json FROM books "
                        "WHERE family=? AND asset=? AND spot IS NOT NULL", con, params=(fam, asset))
        if b.empty: continue
        b["direction"] = b.direction.fillna("")
        days_logged |= set(b.event_date.unique())
        if fam == "above":
            b["hours"] = (b.resolve_ts - b.ts) / 3600; b["pct"] = (b.spot - b.strike) / b.strike * 100
        else:
            b["hours"] = (b.event_date.map(wend) - b.ts) / 3600
            b["pct"] = np.where(b.direction == "↑", (b.strike - b.spot) / b.spot * 100, (b.spot - b.strike) / b.spot * 100)
        dd = duckdb.connect(path, read_only=True)
        dcol = "direction" if fam == "hit" else "''"
        out = {(str(ed)[:10], d_, float(k)): int(o) for ed, d_, k, o in
               dd.execute(f"SELECT event_date, {dcol}, strike, outcome FROM markets WHERE outcome IS NOT NULL").fetchall()}
        f = dd.execute(f"""SELECT CAST(m.event_date AS VARCHAR) AS event_date, {('m.direction' if fam == 'hit' else "''")} AS direction, m.strike,
                                  f.ts, f.price, f.size, f.side, (f.token = m.yes_token) AS is_yes
                           FROM fills f JOIN markets m USING (condition_id) WHERE f.ts >= {int(b.ts.min())}""").df(); dd.close()
        f["event_date"] = f.event_date.str[:10]
        f["p_yes"] = np.where(f.is_yes, f.price, 1 - f.price); f["buy_yes"] = np.where(f.is_yes, f.side == "BUY", f.side == "SELL")
        for c in CELLS:
            if c["fam"] != fam: continue
            plo, phi = c["pct"][asset] if isinstance(c["pct"], dict) else c["pct"]
            q = b[b.pct.between(plo, phi) & b.hours.between(*c["hours"])]
            if c["dirs"]: q = q[q.direction == c["dirs"]]
            if q.empty: continue
            name = f"{SHORT[asset]} {c['code']}"
            px_now = q.best_ask if c["side"] == "yes" else 1 - q.best_bid
            first = q[px_now <= c["limit"] + EPS].sort_values("ts").groupby(KEYS).head(1)
            spent = {}
            for r in first.itertuples():
                if c["side"] == "yes": lev = [(p, p * z) for p, z in sorted(json.loads(r.asks_json)) if 0 < p <= c["limit"] + EPS]
                else: lev = [(1 - p, (1 - p) * z) for p, z in sorted(json.loads(r.bids_json), reverse=True) if 1 - p <= c["limit"] + EPS and p < 1]
                rem = a.cap - spent.get(r.event_date, 0.0); cost = shares = 0.0
                for px, usd in lev:
                    if rem <= EPS: break
                    take = min(usd, rem); cost += take; shares += take / px; rem -= take
                if cost <= 0: continue
                spent[r.event_date] = spent.get(r.event_date, 0.0) + cost
                o = out.get((r.event_date, r.direction, float(r.strike)))
                win = None if o is None else (o if c["side"] == "yes" else 1 - o)
                trades.append(dict(cell=name, event=r.event_date, market=f"{r.direction}{r.strike:g}", entry=datetime.fromtimestamp(r.ts, NY).strftime("%m-%d %H:%M"),
                                   pct=round(r.pct, 2), hrs=round(r.hours, 1), vwap=round(cost / shares, 4), usd=cost, win=win,
                                   pnl=None if win is None else shares * win - cost))
            # maker ceiling
            w = q.groupby(KEYS).agg(t0=("ts", "min"), t1=("ts", "max")).reset_index()
            g = f.merge(w, on=KEYS); g = g[(g.ts >= g.t0) & (g.ts <= g.t1 + 120)]
            if c["side"] == "yes": g = g[~g.buy_yes & (g.p_yes <= c["limit"] + EPS)].assign(px=lambda x: x.p_yes)
            else: g = g[g.buy_yes & (g.p_yes >= 1 - c["limit"] - EPS)].assign(px=lambda x: 1 - x.p_yes)
            if len(g):
                o = np.array([out.get((ed, d_, float(k)), np.nan) for ed, d_, k in zip(g.event_date, g.direction, g.strike)])
                g = g.assign(win=o if c["side"] == "yes" else 1 - o).dropna(subset=["win"])
                g["usd"] = g["size"] * g.px; g["pnl"] = g["size"] * (g.win - g.px)
                makers.append(g.assign(cell=name)[["cell", "event_date", "usd", "pnl", "px"]])

t = pd.DataFrame(trades)
if t.empty: raise SystemExit("no shadow trades found")
days = sorted(days_logged); order = [f"{s} {c['code']}" for c in CELLS for s in ("ETH", "BTC")]
order = [o for o in order if o in set(t.cell)]
res = t.dropna(subset=["pnl"])
print(f"##### shadow review: {len(days)} event-days logged ({days[0]} .. {days[-1]}), cap ${a.cap:,.0f} per cell per day, taker paper\n")

print("=== 1. per cell ===")
rows = []
for cell in order:
    x = res[res.cell == cell]; pend = t[(t.cell == cell) & t.pnl.isna()]
    d = x.groupby("event").pnl.sum()
    rows.append(dict(cell=cell, trades=len(x), wins=int((x.win == 1).sum()), losses=int((x.win == 0).sum()), pending=len(pend), days_traded=x.event.nunique(),
                     deployed=round(x.usd.sum()), pnl=round(x.pnl.sum()), roi=f"{x.pnl.sum()/max(x.usd.sum(),1):+.1%}", avg_px=round(np.average(x.vwap, weights=x.usd), 4) if len(x) else None,
                     best_day=round(d.max()) if len(d) else 0, worst_day=round(d.min()) if len(d) else 0))
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 2. P&L by event-day ($; blank = no trade) ===")
piv = res.pivot_table(index="event", columns="cell", values="pnl", aggfunc="sum").reindex(index=[d for d in days if d in set(res.event)], columns=order)
dep = res.groupby("event").usd.sum()
piv["TOTAL"] = piv.sum(axis=1); piv["cum"] = piv.TOTAL.cumsum(); piv["deployed"] = dep
print(piv.round(0).to_string(na_rep=""))

print("\n=== 3. combined ===")
tot = piv.TOTAL
print(f"trades {len(res)} (wins {int((res.win == 1).sum())}, losses {int((res.win == 0).sum())}), pending {int(t.pnl.isna().sum())}; "
      f"deployed ${res.usd.sum():,.0f}; P&L ${res.pnl.sum():+,.0f} = {res.pnl.sum()/res.usd.sum():+.2%} on capital deployed")
print(f"days traded {len(tot)}: up {int((tot > 0).sum())}, down {int((tot < 0).sum())}; best ${tot.max():+,.0f}, worst ${tot.min():+,.0f}; "
      f"largest capital out on one day ${dep.max():,.0f} (avg ${dep.mean():,.0f}) -> P&L / peak daily capital {res.pnl.sum()/dep.max():+.1%} over the period")
grp = {"sell-the-downside (carry + sellDn)": [o for o in order if o.endswith("carry") or o.endswith("sellDn")],
       "sell-the-upside (sellUp)": [o for o in order if o.endswith("sellUp")], "buy touch (buyT)": [o for o in order if o.endswith("buyT")]}
for k, cols in grp.items():
    x = res[res.cell.isin(cols)]
    if x.empty: continue
    d = x.groupby("event").pnl.sum(); ld = x[x.win == 0].groupby("event").cell.nunique()
    print(f"  {k:36s} P&L ${x.pnl.sum():+8,.0f} on ${x.usd.sum():>9,.0f} ({x.pnl.sum()/x.usd.sum():+.1%}); losing days {int((d < 0).sum())}/{len(d)}; "
          f"days on which 2+ of its cells lost: {int((ld >= 2).sum())}")

if makers:
    m = pd.concat(makers)
    print("\n=== 4. maker ceiling (every taker print that hit a resting order inside the cell's window) ===")
    mm = m.groupby("cell").agg(prints=("usd", "size"), usd=("usd", "sum"), pnl=("pnl", "sum")).reindex(order).dropna()
    mm["usd_per_day"] = (mm.usd / len(days)).round(0); mm["roi"] = (mm.pnl / mm.usd).map("{:+.1%}".format)
    mm["worst_day"] = m.groupby(["cell", "event_date"]).pnl.sum().groupby("cell").min().reindex(mm.index).round(0)
    print(mm.round(0).to_string())
    print(f"all cells: ${m.usd.sum():,.0f} of flow, maker P&L ${m.pnl.sum():+,.0f} ({m.pnl.sum()/m.usd.sum():+.1%}); a real maker captures a fraction of this")

pend = t[t.pnl.isna()]
if len(pend): print("\npending (not yet resolved in the fills DBs):"); print(pend[["cell", "event", "market", "entry", "vwap", "usd"]].round(4).to_string(index=False))
if a.trades:
    print("\n=== every trade ==="); print(t.assign(usd=t.usd.round(0), pnl=t.pnl.round(2)).sort_values(["event", "cell"]).to_string(index=False))
