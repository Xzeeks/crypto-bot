#!/usr/bin/env python3
"""c16_vol_scaled.py — filter v3 candidate: distance measured in units of expected movement (z), tested with rotating splits.

    python research/scripts/c16_vol_scaled.py --asset ethereum [--data-dir research/data] [--seen-start 2023-12-03]

Needs c12_<a>_R.pkl / _F.pkl built over the long history (c14 backfill, then c12).
  z = distance % / expected move over the rule's horizon, where expected move = 30-day realized vol (as known at the start
      of the entry day) scaled to the horizon: carry 24h, sellDn 6h, sellUp 9h, buyT 18h.  z = 3 means "3 typical moves away".
  0. DIAGNOSTICS: loss rate by z-bucket x year (is risk stable once distance is vol-scaled?) next to loss rate by fixed
     distance x year; and what Polymarket charged in 2026 by fixed distance on low-vol vs high-vol days (does the market
     price vol?).
  1. z threshold swept one value at a time on top of v1, unseen vs seen years.
  2. ROTATING SPLITS over a grid that includes z: tune on the other years, score the held-out year, next to none/v1/v2/v3.
  3. v3 candidate, fixed below, by year and as one selling book next to none/v1/v2.
Prices: per fixed distance, from the 2026 Polymarket fills that satisfy the same filter (fallback: all fills at that
distance). Every $ figure for years before 2026 is a simulation at those prices.
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
HOURS = {"carry": 24, "sellDn": 6, "sellUp": 9, "buyT": 18}      # the horizon each rule is exposed for at its standard entry
def add_z(df, distcol):
    sig = df.rvv.astype(float) / np.sqrt(365.0) * np.sqrt(df.cell.map(HOURS).astype(float) / 24.0)      # expected move over the horizon, %
    df["sig"] = sig; df["z"] = df[distcol].astype(float) / sig
add_z(R, "dist")
if len(F) and "cell" in F: add_z(F, "pct")
SELL = {"carry": True, "sellDn": True, "sellUp": True, "buyT": False}
CELLS = [c for c in ("carry", "sellDn", "sellUp", "buyT") if (R.cell == c).any()]
V1 = {"carry": dict(r24_min=-2, trend_min=0), "sellDn": dict(r24_min=-2), "sellUp": dict(r24_min=-4, r24_max=2), "buyT": dict(abs_min=4, skip=("Sat",))}
V2 = {"carry": dict(r24_min=-2, trend_min=0, dist_min=6, skip=("Mon", "Tue")), "sellDn": dict(r24_min=-2, dist_min=5, skip=("Sun",)),
      "sellUp": dict(r24_min=-4, r24_max=2, skip=("Mon",), dist_min=4 if a.asset == "ethereum" else 3), "buyT": dict(abs_min=4, skip=("Sat",))}
# filter v3 candidate (2026-11-28): v1's conditions, with distance measured in typical moves (z) instead of a fixed % floor.
# Same thresholds on both assets, taken from plateaus (neighbouring z values give similar results). Carry's Mon/Tue rule is dropped
# (not supported on 2020-2023); Sunday (sellDn) and Monday (sellUp) stay; buyT is unchanged.
V3 = {"carry": dict(r24_min=-2, trend_min=0, z_min=2.5), "sellDn": dict(r24_min=-2, skip=("Sun",), z_min=3.5),
      "sellUp": dict(r24_min=-4, r24_max=2, skip=("Mon",), z_min=2.5), "buyT": dict(abs_min=4, skip=("Sat",))}
ZGRID = {"carry": [NONE, 1.5, 2.0, 2.5, 3.0], "sellDn": [NONE, 2.5, 3.0, 3.5, 4.0, 4.5], "sellUp": [NONE, 2.0, 2.5, 3.0, 3.5], "buyT": [NONE, 1.5, 2.0, 2.5]}
GRID = {
    "carry":  dict(r24_min=[NONE, -2], trend_min=[NONE, 0], z_min=ZGRID["carry"], dist_min=[5, 6], skip=[(), ("Mon", "Tue")]),
    "sellDn": dict(r24_min=[NONE, -3, -2], rng_max=[NONE, 6, 4], z_min=ZGRID["sellDn"], dist_min=[4, 5], skip=[(), ("Sun",)]),
    "sellUp": dict(r24_min=[NONE, -4], r24_max=[NONE, 3, 2], rng_max=[NONE, 4, 3], z_min=ZGRID["sellUp"], dist_min=[3, 4], skip=[(), ("Mon",)]),
    "buyT":   dict(abs_min=[NONE, 3, 4], skip=[(), ("Sat",)], z_max=ZGRID["buyT"]),
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
    if p.get("z_min") is not NONE: m &= df.z.values >= p["z_min"]
    if p.get("z_max") is not NONE: m &= df.z.values <= p["z_max"]
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

FILTERS = [("none", {c_: {} for c_ in CELLS}), ("v1", V1), ("v2", V2)] + ([("v3", V3)] if V3 else [])
full_years = lambda Rc: [y for y, n_ in Rc.groupby("year").D.nunique().items() if n_ >= a.min_year_days]

# ---------- 0. diagnostics ----------
print("\n################ 0. DIAGNOSTICS ################")
for cell in CELLS:
    Rc = RC[cell]; kind = "LOSS" if SELL[cell] else "HIT"
    edges = [0] + [round(float(v_), 2) for v_ in Rc.z.quantile([0.2, 0.4, 0.6, 0.8])] + [99]
    t = (Rc.assign(zb=pd.cut(Rc.z, edges)).groupby(["zb", "year"], observed=True).ev.mean() * 100).round(1).unstack()
    print(f"\n================ {a.asset} {cell}: {kind} rate % by z-bucket x year (stable across a row = vol-scaling captures the risk) ================"); print(t.to_string(na_rep=""))
    t = (Rc.groupby(["dist", "year"]).ev.mean() * 100).round(1).unstack(); print(f"-- same by fixed distance % x year --"); print(t.to_string(na_rep=""))
    y = FC[cell]
    if y is not None:
        med = y.groupby("D").rvv.first().median(); y = y.assign(vol=np.where(y.rvv > med, "high-vol days", "low-vol days"), d=y.pct.round().astype(int))
        g = y.groupby(["d", "vol"]).apply(lambda v_: pd.Series({"priced%": round(((1 - wavg(v_.q.values, v_.usd.values)) if SELL[cell] else wavg(v_.q.values, v_.usd.values)) * 100, 2), "fill-days": v_.D.nunique()}), include_groups=False).unstack()
        print(f"-- what Polymarket charged in 2026 by fixed distance %, days split at 30d vol {med:.0f}% (similar columns = the market does not price vol) --"); print(g.to_string(na_rep=""))

# ---------- 1. z swept one value at a time ----------
print("\n################ 1. z THRESHOLD, ONE VALUE AT A TIME (on top of v1), unseen vs seen years ################")
for cell in CELLS:
    key = "z_min" if SELL[cell] else "z_max"; rows = []
    for v in ZGRID[cell]:
        t = trades(cell, dict(V1[cell], **{key: v})); r = {key: show(v), "fdays": t.attrs["fdays"]}
        yr = t.groupby("year").pnl.sum().reindex(full_years(RC[cell])).fillna(0)
        for era in ("unseen", "seen"):
            e = t[t.era == era]; r.update({f"{era} trades": len(e), f"{era} rate%": rate(e), f"{era} edge": edge(cell, e), f"{era} roi%": roi(e), f"{era} roi% 1pt": roi(e, "pnl1")})
        r["years +"] = f"{int((yr > 0).sum())}/{len(yr)}"; r["worst year $"] = round(yr.min()); rows.append(r)
    print(f"\n================ {a.asset} {cell} ================"); print(pd.DataFrame(rows).to_string(index=False, na_rep=""))

# ---------- 2. rotating splits ----------
print("\n################ 2. ROTATING SPLITS over the z grid: tune on all other years, score the held-out year ################")
print("Years marked * are before the original design sample. v3 was chosen after looking at EVERY year, so only the 'tuned' column is out of sample here.")
summ = []
for cell in CELLS:
    years = full_years(RC[cell])
    def per_year(p):
        t = trades(cell, p); g = t.groupby("year")
        return g.pnl.sum().reindex(years).fillna(0.0).values, g.size().reindex(years).fillna(0).values, g.ev.sum().reindex(years).fillna(0).astype(float).values, t.attrs["fdays"]
    keys = list(GRID[cell]); cand = []
    for combo in itertools.product(*[GRID[cell][k_] for k_ in keys]):
        p = dict(zip(keys, combo)); pnl, n, ev, fd = per_year(p)
        if fd >= a.min_fdays and not np.isnan(pnl).any(): cand.append((p, pnl, n, ev))
    if not cand: print(f"\n{a.asset} {cell}: no candidate has enough Polymarket fill-days to price — skipped"); continue
    PNL = np.array([c_[1] for c_ in cand]); N = np.array([c_[2] for c_ in cand]); EV = np.array([c_[3] for c_ in cand])
    fixed = {nm: per_year(FS[cell]) for nm, FS in FILTERS}
    pct = lambda x_, n_: round(x_ / n_ * 100, 2) if n_ > 0 else np.nan
    rows, picks = [], []
    for j, y in enumerate(years):
        w = int(np.argmax(PNL.sum(axis=1) - PNL[:, j])); p = cand[w][0]; picks.append(p)
        r = {"held-out": f"{y}" + ("*" if date(y, 7, 1) < SEEN else ""), "tuned filter": ", ".join(f"{k_}={show(v_)}" for k_, v_ in p.items() if v_ not in (NONE, ())) or "no filter",
             "tuned n": int(N[w, j]), "tuned rate%": pct(EV[w, j], N[w, j]), "tuned roi%": pct(PNL[w, j], N[w, j] * STAKE), "tuned $": round(PNL[w, j])}
        for nm, (pn, nn, evv, _) in fixed.items(): r.update({f"{nm} n": int(nn[j]), f"{nm} rate%": pct(evv[j], nn[j]), f"{nm} roi%": pct(pn[j], nn[j] * STAKE), f"{nm} $": round(pn[j])})
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"\n================ {a.asset} {cell}: {len(cand)} priced candidates, {len(years)} held-out years ================"); print(df.to_string(index=False, na_rep=""))
    names = ["tuned"] + [nm for nm, _ in FILTERS]
    print(f"held-out years (of {len(years)}) with positive P&L — " + ", ".join(f"{nm}: {int((df[nm + ' $'] > 0).sum())}" for nm in names) + ";  total $ — " + ", ".join(f"{nm}: {df[nm + ' $'].sum():+,.0f}" for nm in names))
    print("how often each setting was chosen across the folds: " + "; ".join(f"{k_}: " + ", ".join(f"{show(v_)}×{c_}" for v_, c_ in Counter(p[k_] for p in picks).most_common()) for k_ in keys))
    summ.append(dict({"cell": cell, "years": len(years)}, **{f"{nm} avg roi%": round(df[nm + " roi%"].mean(), 2) for nm in names}, **{f"{nm} worst yr $": round(df[nm + " $"].min()) for nm in names}))
if summ: print("\n-- 2 summary --"); print(pd.DataFrame(summ).to_string(index=False))

# ---------- 3. the candidates by year, and the selling book ----------
print("\n################ 3. FILTERS BY YEAR (loss rates are facts; $ are a simulation at 2026 prices) ################")
for cell in CELLS:
    T = {nm: trades(cell, FS[cell]) for nm, FS in FILTERS}; rows = []
    def line(lab, f):
        r = {"period": lab}
        for nm in T:
            k_ = f(T[nm]); r.update({f"{nm} n": len(k_), f"{nm} rate%": rate(k_), f"{nm} roi%": roi(k_)})
            if nm == FILTERS[-1][0]: r.update({f"{nm} priced%": priced(cell, k_), f"{nm} roi% 1pt": roi(k_, "pnl1"), f"{nm} $": round(k_.pnl.sum()) if k_.pnl.notna().any() else np.nan})
        return r
    for y in sorted(T["none"].year.unique()): rows.append(line(str(y), lambda t, y=y: t[t.year == y]))
    for era in ("unseen", "seen"): rows.append(line(f"ALL {era.upper()}", lambda t, e=era: t[t.era == e]))
    rows.append(line("ALL YEARS", lambda t: t))
    print(f"\n================ {a.asset} {cell} ({'LOSS' if SELL[cell] else 'HIT'} rate) ================"); print(pd.DataFrame(rows).to_string(index=False, na_rep=""))
sellc = [c for c in ("carry", "sellDn", "sellUp") if c in CELLS]
print(f"\n################ the {len(sellc)} selling rules as one book (SIMULATION at 2026 prices) ################")
rows, byyear = [], {}
for name, FS in FILTERS:
    tt = pd.concat([trades(c_, FS[c_]) for c_ in sellc]); byyear[name] = tt.groupby("year").pnl.sum().round(0)
    for lab, e in (("unseen", tt[tt.era == "unseen"]), ("seen", tt[tt.era == "seen"]), ("all", tt)):
        if not len(e) or not e.pnl.notna().any(): continue
        d = e.groupby("D").pnl.sum().sort_index(); n = e.groupby("D").size() * STAKE
        mon = d.groupby(np.array([x_.year * 100 + x_.month for x_ in d.index])).sum()
        rows.append({"filter": name, "years": lab, "days traded": len(d), "avg $ at work": round(n.mean()), "total staked": round(n.sum()), "pnl": round(d.sum()),
                     "roi%": round(d.sum() / n.sum() * 100, 2), "roi% 1pt": roi(e, "pnl1"), "loss%": rate(e), "losing days": int((d < 0).sum()), "worst day": round(d.min()),
                     "maxDD": round(maxdd(d)), "months +": f"{int((mon > 0).sum())}/{len(mon)}", "worst month": round(mon.min())})
if rows:
    print(pd.DataFrame(rows).to_string(index=False)); print("\n-- P&L by year ($) --"); print(pd.DataFrame(byyear).fillna(0).to_string())
