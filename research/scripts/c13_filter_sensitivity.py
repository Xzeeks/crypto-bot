#!/usr/bin/env python3
"""c13_filter_sensitivity.py — how sensitive is each rule to its filter thresholds, and is there a better set than filter v1?

    python research/scripts/c13_filter_sensitivity.py --asset ethereum [--data-dir research/data]

Needs the tables saved by c12_conditioning.py (c12_<a>_R.pkl = one row per simulated trade over ~3 years with its entry
conditions; c12_<a>_F.pkl = Polymarket fills in each cell with the same conditions).

For every candidate filter:
  kept%        share of the rule's days that still trade
  loss%        event rate on kept trades, split TRAIN (through 2025) / TEST (2026). Loss rate for carry/sellDn/sellUp, hit rate for buyT.
  fdays / px   Polymarket fill-days that satisfy the filter, and the $-weighted price paid on them (per distance in the simulation).
               Candidates with fewer than --min-fdays fill-days are shown as 'thin' and never ranked: their price is not known.
  roi% / pnl   SIMULATION, $1,000 per kept trade at those prices; pnl split TRAIN / TEST; roi if every price were 1 point worse.
Sections per rule: (1) filter v1 and no filter; (2) one threshold varied at a time, the rest held at v1 — look for plateaus,
not peaks; (3) the full grid ranked by TRAIN pnl, with TEST shown alongside — a candidate is only interesting if TEST holds up.
"""
import argparse, itertools, os
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
ap = argparse.ArgumentParser(); ap.add_argument("--asset", default="ethereum"); ap.add_argument("--data-dir", default="research/data")
ap.add_argument("--min-fdays", type=int, default=15); ap.add_argument("--top", type=int, default=8)
a = ap.parse_args()
pre = {"ethereum": "eth", "bitcoin": "btc"}[a.asset]
R = pd.read_pickle(os.path.join(a.data_dir, f"c12_{pre}_R.pkl")); F = pd.read_pickle(os.path.join(a.data_dir, f"c12_{pre}_F.pkl"))
R["year"] = [d.year for d in R.D]; STAKE = 1000.0
SELL = {"carry": True, "sellDn": True, "sellUp": True, "buyT": False}
NONE = None

def mask(df, cell, p, distcol):
    """p: dict of thresholds; None = not applied."""
    m = np.ones(len(df), dtype=bool); r = df.r24v.values; t = df.trendv.values; g = df.rngv.values; dow = df.dow.values; d = df[distcol].values
    if p.get("r24_min") is not NONE: m &= r >= p["r24_min"]
    if p.get("r24_max") is not NONE: m &= r <= p["r24_max"]
    if p.get("abs_min") is not NONE: m &= np.abs(r) > p["abs_min"]
    if p.get("trend_min") is not NONE: m &= t >= p["trend_min"]
    if p.get("rng_max") is not NONE: m &= g <= p["rng_max"]
    if p.get("rng_min") is not NONE: m &= g >= p["rng_min"]
    if p.get("skip") : m &= ~np.isin(dow, list(p["skip"]))
    if p.get("only"): m &= np.isin(dow, list(p["only"]))
    if p.get("dist_min") is not NONE: m &= d >= p["dist_min"] - (0.5 if distcol == "pct" else 0)
    if p.get("dist_max") is not NONE: m &= d <= p["dist_max"] + (0.5 if distcol == "pct" else 0)
    return m

def wavg(v, w): return float(np.average(v, weights=w)) if len(v) and w.sum() > 0 else np.nan
def evaluate(cell, p):
    x = Rc[mask(Rc, cell, p, "dist")]; y = Fc[mask(Fc, cell, p, "pct")] if len(Fc) else Fc
    sell = SELL[cell]; tr, te = x[x.year <= 2025], x[x.year >= 2026]
    out = {"kept%": round(x.D.nunique() / max(Rc.D.nunique(), 1) * 100), "trades": len(x),
           "loss% train": round(tr.ev.mean() * 100, 2) if len(tr) else np.nan, "loss% test": round(te.ev.mean() * 100, 2) if len(te) else np.nan,
           "fdays": y.D.nunique() if len(y) else 0}
    if out["fdays"] < a.min_fdays or len(x) == 0:
        out.update({"px": np.nan, "roi%": np.nan, "roi% 1pt": np.nan, "pnl": np.nan, "pnl train": np.nan, "pnl test": np.nan, "worst day": np.nan, "maxDD": np.nan, "thin": True}); return out
    dists = sorted(x.dist.unique()); dd = np.array(dists, dtype=float); near = dd[np.abs(y.pct.values[:, None] - dd[None, :]).argmin(axis=1)]
    base = wavg(y.q, y.usd); tab = {}
    for d_ in dists:
        m_ = near == d_
        tab[d_] = wavg(y.q[m_], y.usd[m_]) if m_.sum() >= 20 and y.usd[m_].sum() > 0 else base
    q = np.clip(x.dist.map(tab).values, 0.001, 0.999); win = (1 - x.ev.values.astype(float)) if sell else x.ev.values.astype(float)
    pnl = STAKE * (win / q - 1); pnl1 = STAKE * (win / np.clip(q + 0.01, 0.001, 0.999) - 1)
    dly = pd.Series(pnl).groupby(x.D.values).sum(); eq = dly.cumsum(); yr = x.year.values
    out.update({"px": round(base, 4), "roi%": round(pnl.mean() / STAKE * 100, 2), "roi% 1pt": round(pnl1.mean() / STAKE * 100, 2), "pnl": round(pnl.sum()),
                "pnl train": round(pnl[yr <= 2025].sum()), "pnl test": round(pnl[yr >= 2026].sum()), "worst day": round(dly.min()), "maxDD": round((eq - eq.cummax()).min()), "thin": False})
    return out

V1 = {"carry": dict(r24_min=-2, trend_min=0), "sellDn": dict(r24_min=-2), "sellUp": dict(r24_min=-4, r24_max=2), "buyT": dict(abs_min=4, skip=("Sat",))}
ONE = {  # one threshold at a time (others at v1): (label, key, values)
    "carry":  [("24h move no worse than", "r24_min", [NONE, -5, -4, -3, -2, -1, 0]), ("price vs 30d average at least (%)", "trend_min", [NONE, -5, 0, 3, 6]),
               ("nearest distance traded (%)", "dist_min", [5, 6, 7]), ("skip event days", "skip", [(), ("Mon",), ("Mon", "Tue")]), ("only event days", "only", [(), ("Sat", "Sun")])],
    "sellDn": [("24h move no worse than", "r24_min", [NONE, -5, -4, -3, -2, -1, 0]), ("day's range so far at most (%)", "rng_max", [NONE, 6, 4, 3, 2]),
               ("price vs 30d average at least (%)", "trend_min", [NONE, -5, 0]), ("nearest distance traded (%)", "dist_min", [4, 5, 6]),
               ("skip event days", "skip", [(), ("Sun",), ("Sun", "Thu")]), ("only event days", "only", [(), ("Sat",)])],
    "sellUp": [("24h move no better than", "r24_max", [NONE, 4, 3, 2, 1, 0]), ("24h move no worse than", "r24_min", [NONE, -6, -4, -3, -2]),
               ("day's range so far at most (%)", "rng_max", [NONE, 6, 4, 3, 2]), ("nearest distance traded (%)", "dist_min", [3, 4, 5]),
               ("skip event days", "skip", [(), ("Mon",), ("Mon", "Wed")]), ("only event days", "only", [(), ("Fri", "Sat", "Sun")])],
    "buyT":   [("24h move larger than (either way)", "abs_min", [NONE, 1, 2, 3, 4, 5, 6]), ("day's range so far at least (%)", "rng_min", [NONE, 1, 2, 3]),
               ("skip event days", "skip", [(), ("Sat",), ("Sat", "Sun")]), ("farthest distance traded (%)", "dist_max", [5, 4, 3])],
}
GRID = {
    "carry":  dict(r24_min=[NONE, -4, -3, -2, -1], trend_min=[NONE, -5, 0, 3], dist_min=[5, 6, 7], skip=[(), ("Mon", "Tue")]),
    "sellDn": dict(r24_min=[NONE, -4, -3, -2, -1], rng_max=[NONE, 6, 4, 3], trend_min=[NONE, 0], dist_min=[4, 5], skip=[(), ("Sun",)]),
    "sellUp": dict(r24_max=[NONE, 4, 3, 2, 1], r24_min=[NONE, -4, -2], rng_max=[NONE, 6, 4, 3], dist_min=[3, 4], skip=[(), ("Mon",)]),
    "buyT":   dict(abs_min=[NONE, 2, 3, 4, 5], rng_min=[NONE, 2, 3], skip=[(), ("Sat",), ("Sat", "Sun")], dist_max=[5, 4]),
}
COLS = ["kept%", "trades", "loss% train", "loss% test", "fdays", "px", "roi%", "roi% 1pt", "pnl", "pnl train", "pnl test", "worst day", "maxDD"]
show = lambda v: "none" if v is NONE else ("-" if v == () else ("+".join(v) if isinstance(v, tuple) else str(v)))
def fmt(rows):
    df = pd.DataFrame(rows)
    for c_ in ("px", "roi%", "roi% 1pt", "pnl", "pnl train", "pnl test", "worst day", "maxDD"):
        df[c_] = [("thin" if t_ else v_) for v_, t_ in zip(df[c_], df["thin"])] if c_ == "px" else df[c_]
    return df.drop(columns=["thin"]).to_string(index=False, na_rep="")

print(f"##### {a.asset}: {R.D.nunique()} days ({R.D.min()} .. {R.D.max()}); TRAIN = through 2025, TEST = 2026; ${STAKE:,.0f} per trade; "
      f"'thin' = fewer than {a.min_fdays} Polymarket fill-days, price unknown")
for cell in ("carry", "sellDn", "sellUp", "buyT"):
    Rc = R[R.cell == cell].reset_index(drop=True); Fc = F[F.cell == cell].reset_index(drop=True) if len(F) else F
    kind = "LOSS rate" if SELL[cell] else "HIT rate (higher is better for the buyer)"
    print(f"\n================ {a.asset} {cell}: loss% columns show the {kind} ================")
    print("\n-- 1. baseline --")
    print(fmt([dict({"filter": "none"}, **evaluate(cell, {})), dict({"filter": "v1"}, **evaluate(cell, V1[cell]))]))
    print("\n-- 2. one threshold at a time (everything else at v1) --")
    for label, key, vals in ONE[cell]:
        rows = []
        for v in vals:
            p = dict(V1[cell]); p[key] = v
            if key == "only" and v: p.pop("skip", None)
            rows.append(dict({label: show(v) + (" *" if V1[cell].get(key, NONE if key not in ("skip", "only") else ()) == v else "")}, **evaluate(cell, p)))
        print(fmt(rows) + "\n")
    print(f"-- 3. full grid, top {a.top} by TRAIN pnl (thin candidates excluded); check that TEST holds up --")
    keys = list(GRID[cell]); res = []
    for combo in itertools.product(*[GRID[cell][k_] for k_ in keys]):
        p = dict(zip(keys, combo)); e = evaluate(cell, p)
        if not e["thin"]: res.append(dict({k_: show(v_) for k_, v_ in p.items()}, **e))
    if res:
        g = pd.DataFrame(res).sort_values("pnl train", ascending=False)
        print(g.head(a.top).drop(columns=["thin"]).to_string(index=False))
        pos = g[(g["pnl train"] > 0) & (g["pnl test"] > 0)]
        print(f"candidates priced: {len(g)}; positive in BOTH train and test: {len(pos)}; of the top {a.top} by train, positive in test: {int((g.head(a.top)['pnl test'] > 0).sum())}")
        rb = g[(g["pnl train"] > 0) & (g["pnl test"] > 0) & (g["roi% 1pt"] > 0)].sort_values("roi% 1pt", ascending=False)
        if len(rb): print("most robust (positive in train AND test AND still positive if prices are 1 point worse), by roi% 1pt:"); print(rb.head(5).drop(columns=["thin"]).to_string(index=False))
        else: print("no candidate is positive in train, in test, and at prices 1 point worse")
    else:
        print("no candidate has enough Polymarket fill-days to price")
