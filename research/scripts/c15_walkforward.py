#!/usr/bin/env python3
"""c15_walkforward.py — out-of-sample checks of filter v1 / v2 on a longer price history, and rotating train/test splits.

    python research/scripts/c15_walkforward.py --asset ethereum [--data-dir research/data] [--seen-start 2023-12-03]

Run AFTER c14_klines_backfill.py (older klines) and a rerun of c12_conditioning.py (rebuilds c12_<a>_R.pkl / _F.pkl over the
longer history). Filters v1 and v2 were both chosen looking only at days from --seen-start onward (the original 3-year
sample), so every earlier day is UNSEEN: the filters are applied there exactly as fixed, nothing is tuned.

  A. FROZEN filters by year: loss rate with no filter, on v1/v2 kept trades and on the trades each filter drops.
     Loss rates use Binance prices only = FACT. 'priced%' is the loss rate today's Polymarket prices pay for (1 - price for
     the selling rules, price for buyT); 'edge' = priced% - loss% (buyT: hit% - priced%). roi% is a SIMULATION at 2026 prices.
  B. v2's additions one at a time on top of v1 (distance, skipped days), unseen vs seen years, plus the loss rate by day
     of week on v1-kept trades — the direct test of whether the skipped days are bad outside the years they were picked on.
  C. The three selling rules as one book: no filter / v1 / v2, unseen vs seen years.
  D. ROTATING SPLITS (leave one year out): tune the c13 grid on all other years, score the held-out year; compare with
     v1 / v2 / no filter in that same year, and count how often each threshold value is chosen (stable = real, scattered = noise).
Polymarket prices exist only for 2026, so every roi/pnl prices old years at 2026 levels — older, more volatile years would
likely have been priced differently. Lean on the loss rates and the kept-vs-skipped gaps; treat $ figures as indicative.
"""
import argparse, itertools, os
from collections import Counter
from datetime import date
import numpy as np, pandas as pd
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)
ap = argparse.ArgumentParser(); ap.add_argument("--asset", default="ethereum"); ap.add_argument("--data-dir", default="research/data")
ap.add_argument("--seen-start", default="2023-12-03", help="first day of the sample v1/v2 were designed on")
ap.add_argument("--min-fdays", type=int, default=15); ap.add_argument("--min-year-days", type=int, default=200)
a = ap.parse_args()
pre = {"ethereum": "eth", "bitcoin": "btc"}[a.asset]
R = pd.read_pickle(os.path.join(a.data_dir, f"c12_{pre}_R.pkl")); F = pd.read_pickle(os.path.join(a.data_dir, f"c12_{pre}_F.pkl"))
SEEN = date.fromisoformat(a.seen_start); STAKE = 1000.0; NONE = None
R["year"] = [d.year for d in R.D]; R["era"] = np.where(np.array([d < SEEN for d in R.D]), "unseen", "seen")
SELL = {"carry": True, "sellDn": True, "sellUp": True, "buyT": False}
CELLS = [c for c in ("carry", "sellDn", "sellUp", "buyT") if (R.cell == c).any()]
V1 = {"carry": dict(r24_min=-2, trend_min=0), "sellDn": dict(r24_min=-2), "sellUp": dict(r24_min=-4, r24_max=2), "buyT": dict(abs_min=4, skip=("Sat",))}
V2 = {"carry": dict(r24_min=-2, trend_min=0, dist_min=6, skip=("Mon", "Tue")), "sellDn": dict(r24_min=-2, dist_min=5, skip=("Sun",)),
      "sellUp": dict(r24_min=-4, r24_max=2, skip=("Mon",), dist_min=4 if a.asset == "ethereum" else 3), "buyT": dict(abs_min=4, skip=("Sat",))}
GRID = {   # same grid as c13_filter_sensitivity.py
    "carry":  dict(r24_min=[NONE, -4, -3, -2, -1], trend_min=[NONE, -5, 0, 3], dist_min=[5, 6, 7], skip=[(), ("Mon", "Tue")]),
    "sellDn": dict(r24_min=[NONE, -4, -3, -2, -1], rng_max=[NONE, 6, 4, 3], trend_min=[NONE, 0], dist_min=[4, 5], skip=[(), ("Sun",)]),
    "sellUp": dict(r24_max=[NONE, 4, 3, 2, 1], r24_min=[NONE, -4, -2], rng_max=[NONE, 6, 4, 3], dist_min=[3, 4], skip=[(), ("Mon",)]),
    "buyT":   dict(abs_min=[NONE, 2, 3, 4, 5], rng_min=[NONE, 2, 3], skip=[(), ("Sat",), ("Sat", "Sun")], dist_max=[5, 4]),
}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
show = lambda v: "none" if v is NONE else ("-" if v == () else ("+".join(v) if isinstance(v, tuple) else str(v)))

def mask(df, p, distcol):
    m = np.ones(len(df), dtype=bool); r = df.r24v.values; t = df.trendv.values; g = df.rngv.values; dow = df.dow.values; d = df[distcol].values
    if p.get("r24_min") is not NONE: m &= r >= p["r24_min"]
    if p.get("r24_max") is not NONE: m &= r <= p["r24_max"]
    if p.get("abs_min") is not NONE: m &= np.abs(r) > p["abs_min"]
    if p.get("trend_min") is not NONE: m &= t >= p["trend_min"]
    if p.get("rng_max") is not NONE: m &= g <= p["rng_max"]
    if p.get("rng_min") is not NONE: m &= g >= p["rng_min"]
    if p.get("skip"): m &= ~np.isin(dow, list(p["skip"]))
    if p.get("only"): m &= np.isin(dow, list(p["only"]))
    if p.get("dist_min") is not NONE: m &= d >= p["dist_min"] - (0.5 if distcol == "pct" else 0)
    if p.get("dist_max") is not NONE: m &= d <= p["dist_max"] + (0.5 if distcol == "pct" else 0)
    return m
def wavg(v, w): return float(np.average(v, weights=w)) if len(v) and np.sum(w) > 0 else np.nan

RC, FC, DISTS, ALLTAB, ALLMAX = {}, {}, {}, {}, {}
for cell in CELLS:
    RC[cell] = R[R.cell == cell].reset_index(drop=True)
    FC[cell] = F[F.cell == cell].reset_index(drop=True) if len(F) and "cell" in F and (F.cell == cell).any() else None
    DISTS[cell] = sorted(RC[cell].dist.unique()); ALLTAB[cell], ALLMAX[cell] = {}, np.nan
    if FC[cell] is not None:
        y = FC[cell]; dd = np.array(DISTS[cell], dtype=float); near = dd[np.abs(y.pct.values[:, None] - dd[None, :]).argmin(axis=1)]
        ALLTAB[cell] = {d_: wavg(y.q.values[near == d_], y.usd.values[near == d_]) for d_ in DISTS[cell] if (near == d_).sum() >= 20}
        ALLMAX[cell] = max(ALLTAB[cell].values()) if ALLTAB[cell] else wavg(y.q.values, y.usd.values)

def price_table(cell, p):
    """Price per distance from the Polymarket fills that satisfy filter p; where a distance has too few such fills, the
       unfiltered price at that distance; failing that, the most expensive known price. Returns (table, fill-days)."""
    y = FC[cell]
    if y is None: return {d_: np.nan for d_ in DISTS[cell]}, 0
    y = y[mask(y, p, "pct")]; tab = {}
    dd = np.array(DISTS[cell], dtype=float); near = dd[np.abs(y.pct.values[:, None] - dd[None, :]).argmin(axis=1)] if len(y) else np.array([])
    for d_ in DISTS[cell]:
        m_ = near == d_
        tab[d_] = wavg(y.q.values[m_], y.usd.values[m_]) if len(y) and m_.sum() >= 20 and y.usd.values[m_].sum() > 0 else ALLTAB[cell].get(d_, ALLMAX[cell])
    return tab, (y.D.nunique() if len(y) else 0)
def trades(cell, p, drop=False):
    """One row per simulated trade the filter keeps (or, with drop=True, the trades it drops), priced at the filter's own price table."""
    Rc = RC[cell]; m = mask(Rc, p, "dist"); x = Rc[~m if drop else m][["D", "year", "era", "dist", "dow", "ev"]].copy()
    tab, fdays = price_table(cell, p)
    q = np.clip(x.dist.map(tab).astype(float).values, 0.001, 0.999); ev = x.ev.values.astype(float); win = (1 - ev) if SELL[cell] else ev
    x["q"] = q; x["pnl"] = STAKE * (win / q - 1); x["pnl1"] = STAKE * (win / np.clip(q + 0.01, 0.001, 0.999) - 1); x.attrs["fdays"] = fdays
    return x
rate = lambda t: round(t.ev.mean() * 100, 2) if len(t) else np.nan
roi = lambda t, col="pnl": round(t[col].mean() / STAKE * 100, 2) if len(t) and t[col].notna().any() else np.nan
def priced(cell, t):
    if not len(t) or not t.q.notna().any(): return np.nan
    return round(((1 - t.q.mean()) if SELL[cell] else t.q.mean()) * 100, 2)
def edge(cell, t):
    pr, rt = priced(cell, t), rate(t)
    return np.nan if pr != pr or rt != rt else round((pr - rt) if SELL[cell] else (rt - pr), 2)
def maxdd(d): eq = d.cumsum(); return float((eq - eq.cummax()).min()) if len(eq) else 0.0

yrs_all = sorted(R.year.unique())
print(f"##### {a.asset}: {R.D.nunique()} days ({R.D.min()} .. {R.D.max()}). UNSEEN = before {SEEN} ({R[R.era == 'unseen'].D.nunique()} days, never used to design a filter); "
      f"SEEN = from {SEEN} ({R[R.era == 'seen'].D.nunique()} days). ${STAKE:,.0f} per trade; all prices are 2026 Polymarket prices.")
if not (R.era == "unseen").any():
    print("!! no days before --seen-start: run c14_klines_backfill.py and rerun c12_conditioning.py first. Sections A-C will only show the seen years.")

# ---------- A. frozen filters by year ----------
print("\n################ A. FROZEN FILTERS BY YEAR (loss rates are facts; priced%/edge/roi use 2026 prices) ################")
verdicts = []
for cell in CELLS:
    kind = "LOSS rate" if SELL[cell] else "HIT rate (higher is better for the buyer)"
    T = {"none": trades(cell, {}), "v1": trades(cell, V1[cell]), "v2": trades(cell, V2[cell])}
    X = {"v1": trades(cell, V1[cell], drop=True), "v2": trades(cell, V2[cell], drop=True)}
    print(f"\n================ {a.asset} {cell}: rates are the {kind}; v2 priced from {T['v2'].attrs['fdays']} fill-days ================")
    rows = []
    def line(lab, f):
        n_ = f(T["none"]); r = {"period": lab, "days": n_.D.nunique(), "none%": rate(n_)}
        for v in ("v1", "v2"):
            k_, s_ = f(T[v]), f(X[v])
            r.update({f"{v} trades": len(k_), f"{v} kept%": rate(k_), f"{v} dropped%": rate(s_)})
        k2 = f(T["v2"]); k1 = f(T["v1"])
        r.update({"v1 roi%": roi(k1), "v2 priced%": priced(cell, k2), "v2 edge": edge(cell, k2), "v2 roi%": roi(k2), "v2 roi% 1pt": roi(k2, "pnl1")})
        return r
    for (era, yr), _ in T["none"].groupby(["era", "year"], sort=False):
        rows.append(line(f"{yr}" + ("" if era == "unseen" else " (seen)"), lambda t, e=era, y=yr: t[(t.era == e) & (t.year == y)]))
    rows.sort(key=lambda r: r["period"])
    for era in ("unseen", "seen"):
        if (T["none"].era == era).any(): rows.append(line(f"ALL {era.upper()}", lambda t, e=era: t[t.era == e]))
    df = pd.DataFrame(rows); print(df.to_string(index=False, na_rep=""))
    u = df[~df.period.str.contains("seen|ALL", regex=True) & (df.days >= a.min_year_days)]
    if len(u):
        better = (lambda k, s: k < s) if SELL[cell] else (lambda k, s: k > s)
        v = dict(cell=cell, years=len(u),
                 v1_kept_beats_dropped=int(sum(better(k, s) for k, s in zip(u["v1 kept%"], u["v1 dropped%"]))),
                 v2_kept_beats_dropped=int(sum(better(k, s) for k, s in zip(u["v2 kept%"], u["v2 dropped%"]))),
                 v2_beats_v1=int(sum(better(k2, k1) for k2, k1 in zip(u["v2 kept%"], u["v1 kept%"]))),
                 v2_edge_positive=int((u["v2 edge"] > 0).sum()), v2_roi_positive=int((u["v2 roi%"] > 0).sum()), v2_roi_1pt_positive=int((u["v2 roi% 1pt"] > 0).sum()))
        verdicts.append(v)
if verdicts:
    print("\n-- A summary: in how many UNSEEN full years did each statement hold (out of 'years') --")
    print(pd.DataFrame(verdicts).to_string(index=False))

# ---------- B. v2's additions one at a time ----------
print("\n################ B. v2's ADDITIONS ONE AT A TIME (on top of v1), unseen vs seen years ################")
print("Dropping the nearest distance lowers the loss rate mechanically (farther is safer) — judge that one on edge/roi. A skipped day is real only if that day is worse in the UNSEEN years too.")
for cell in CELLS:
    extra = {k_: v_ for k_, v_ in V2[cell].items() if V1[cell].get(k_) != v_}
    if not extra: print(f"\n{a.asset} {cell}: v2 = v1, nothing to test"); continue
    variants = [("v1", V1[cell])] + [(f"v1 + {k_}={show(v_)}", dict(V1[cell], **{k_: v_})) for k_, v_ in extra.items()] + ([("v2 (all)", V2[cell])] if len(extra) > 1 else [])
    rows = []
    for lab, p in variants:
        t = trades(cell, p); r = {"filter": lab}
        for era in ("unseen", "seen"):
            e = t[t.era == era]
            r.update({f"{era} trades": len(e), f"{era} rate%": rate(e), f"{era} edge": edge(cell, e), f"{era} roi%": roi(e), f"{era} roi% 1pt": roi(e, "pnl1")})
        rows.append(r)
    print(f"\n================ {a.asset} {cell} ================"); print(pd.DataFrame(rows).to_string(index=False, na_rep=""))
    if "skip" in extra:
        t = trades(cell, {k_: v_ for k_, v_ in V2[cell].items() if k_ != "skip"}); rows = []     # v2 without its day rule: is the skipped day worse?
        for d_ in DOW:
            r = {"event day": d_ + (" (v2 skips)" if d_ in extra["skip"] else "")}
            for era in ("unseen", "seen"):
                e = t[(t.era == era) & (t.dow == d_)]; r.update({f"{era} trades": len(e), f"{era} rate%": rate(e), f"{era} roi%": roi(e)})
            rows.append(r)
        print("-- by day of week, v2 without its day rule --"); print(pd.DataFrame(rows).to_string(index=False, na_rep=""))
        for era in ("unseen", "seen"):
            e = t[t.era == era]
            if not len(e): continue
            by = e.groupby("dow").ev.mean() * 100; rank = by.rank(ascending=not SELL[cell])      # rank 7 = worst day for the rule
            print(f"   {era}: " + ", ".join(f"{d_} is #{int(len(by) + 1 - rank[d_])} worst of {len(by)}" for d_ in extra["skip"] if d_ in by))

# ---------- C. selling book ----------
sellc = [c for c in ("carry", "sellDn", "sellUp") if c in CELLS]
print(f"\n################ C. THE {len(sellc)} SELLING RULES AS ONE BOOK (SIMULATION at 2026 prices) ################")
rows, byyear = [], {}
for name, FS in (("none", {c_: {} for c_ in sellc}), ("v1", V1), ("v2", V2)):
    tt = pd.concat([trades(c_, FS[c_]) for c_ in sellc])
    byyear[name] = tt.groupby("year").pnl.sum().round(0)
    for era in ("unseen", "seen"):
        e = tt[tt.era == era]
        if not len(e) or not e.pnl.notna().any(): continue
        d = e.groupby("D").pnl.sum().sort_index(); n = e.groupby("D").size() * STAKE
        mon = d.groupby(np.array([x_.year * 100 + x_.month for x_ in d.index])).sum()
        rows.append({"filter": name, "years": era, "days traded": len(d), "avg $ at work": round(n.mean()), "total staked": round(n.sum()), "pnl": round(d.sum()),
                     "roi%": round(d.sum() / n.sum() * 100, 2), "roi% 1pt": roi(e, "pnl1"), "loss%": rate(e), "losing days": int((d < 0).sum()), "worst day": round(d.min()),
                     "maxDD": round(maxdd(d)), "months +": f"{int((mon > 0).sum())}/{len(mon)}", "worst month": round(mon.min())})
if rows:
    print(pd.DataFrame(rows).to_string(index=False)); print("\n-- P&L by year ($) --"); print(pd.DataFrame(byyear).fillna(0).to_string())
else: print("no Polymarket fills to price from")

# ---------- D. rotating splits ----------
print("\n################ D. ROTATING SPLITS: tune on all other years, score the held-out year ################")
print("'tuned' = the grid candidate with the highest simulated P&L on the other years. Years marked * are UNSEEN by v1/v2; in the other years v1/v2 have an unfair advantage (they were chosen there).")
summ = []
for cell in CELLS:
    Rc = RC[cell]; ydays = Rc.groupby("year").D.nunique(); years = [y for y in sorted(ydays.index) if ydays[y] >= a.min_year_days]
    if len(years) < 3: print(f"\n{a.asset} {cell}: fewer than 3 full years — skipped"); continue
    def per_year(p):
        t = trades(cell, p); g = t.groupby("year")
        return g.pnl.sum().reindex(years).fillna(0.0).values, g.size().reindex(years).fillna(0).values, g.ev.sum().reindex(years).fillna(0).astype(float).values, t.attrs["fdays"]
    keys = list(GRID[cell]); cand = []
    for combo in itertools.product(*[GRID[cell][k_] for k_ in keys]):
        p = dict(zip(keys, combo)); pnl, n, ev, fd = per_year(p)
        if fd >= a.min_fdays and not np.isnan(pnl).any(): cand.append((p, pnl, n, ev))
    if not cand: print(f"\n{a.asset} {cell}: no grid candidate has enough Polymarket fill-days to price — skipped"); continue
    PNL = np.array([c_[1] for c_ in cand]); N = np.array([c_[2] for c_ in cand]); EV = np.array([c_[3] for c_ in cand])
    fixed = {nm: per_year(p) for nm, p in (("none", {}), ("v1", V1[cell]), ("v2", V2[cell]))}
    pct = lambda x_, n_: round(x_ / n_ * 100, 2) if n_ > 0 else np.nan
    rows, picks = [], []
    for j, y in enumerate(years):
        w = int(np.argmax(PNL.sum(axis=1) - PNL[:, j])); p = cand[w][0]; picks.append(p)
        r = {"held-out": f"{y}" + ("*" if date(y, 7, 1) < SEEN else ""), "tuned filter": ", ".join(f"{k_}={show(v_)}" for k_, v_ in p.items() if v_ not in (NONE, ())) or "no filter",
             "tuned n": int(N[w, j]), "tuned rate%": pct(EV[w, j], N[w, j]), "tuned roi%": pct(PNL[w, j], N[w, j] * STAKE)}
        for nm, (pn, nn, evv, _) in fixed.items():
            r.update({f"{nm} rate%": pct(evv[j], nn[j]), f"{nm} roi%": pct(pn[j], nn[j] * STAKE)})
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"\n================ {a.asset} {cell}: {len(cand)} priced candidates, {len(years)} held-out years ================"); print(df.to_string(index=False, na_rep=""))
    gt = lambda x_, y_: int((df[x_].fillna(-1e9) > df[y_].fillna(-1e9)).sum())
    print(f"held-out years (of {len(years)}) with positive roi — tuned: {int((df['tuned roi%'] > 0).sum())}, v2: {int((df['v2 roi%'] > 0).sum())}, v1: {int((df['v1 roi%'] > 0).sum())}, none: {int((df['none roi%'] > 0).sum())};  "
          f"tuned beats v1 in {gt('tuned roi%', 'v1 roi%')}, tuned beats v2 in {gt('tuned roi%', 'v2 roi%')}, v2 beats v1 in {gt('v2 roi%', 'v1 roi%')}, v1 beats none in {gt('v1 roi%', 'none roi%')}")
    print("how often each setting was chosen across the folds: " + "; ".join(f"{k_}: " + ", ".join(f"{show(v_)}×{c_}" for v_, c_ in Counter(p[k_] for p in picks).most_common()) for k_ in keys))
    summ.append({"cell": cell, "years": len(years), "avg held-out roi% tuned": round(df["tuned roi%"].mean(), 2), "v2": round(df["v2 roi%"].mean(), 2), "v1": round(df["v1 roi%"].mean(), 2), "none": round(df["none roi%"].mean(), 2),
                 "worst year tuned": df["tuned roi%"].min(), "worst year v2": df["v2 roi%"].min(), "worst year v1": df["v1 roi%"].min()})
if summ:
    print("\n-- D summary: average and worst held-out-year roi% --"); print(pd.DataFrame(summ).to_string(index=False))
