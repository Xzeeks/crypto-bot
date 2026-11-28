#!/usr/bin/env python3
"""c18_sizing.py — how much to stake on each selling trade: a loss-probability model, a cautious edge, and sizing rules
tested on the long price history with a daily loss cap.

    python research/scripts/c18_sizing.py [--data-dir research/data] [--bankroll 20000] [--day-cap 0.10] [--kelly 0.10] [--safety 1.75]

Needs c12_<a>_R.pkl / _F.pkl for both assets (c12_conditioning.py over the long history). Selling rules only (carry, sellDn,
sellUp); buy touch is left out. ETH and BTC are pooled: a day's trades on both assets count against one cap.

1. PROBABILITY. A logistic model per rule for "this trade loses", from: distance in typical moves (z), the 24h move in
   typical moves (falls and rises separately), above/below the 30-day average, the day's range so far in typical moves
   (touch rules), and the asset. Every prediction used below is OUT OF SAMPLE BY YEAR: the model that scores a year was
   fitted on the other years.
2. CAUTIOUS EDGE. edge = (1 - price) - safety x predicted loss rate. safety (default 1.75) is roughly the worst year's
   actual/predicted ratio, so the edge is what is left if the year turns out like the worst one.
3. SIZE. Kelly share of bankroll for a bet bought at price p that wins with probability q is (q - p) / (1 - p); the rule
   stakes --kelly (default a tenth) of that, using the cautious q, at most --trade-cap per trade. No stake if the cautious
   edge is not positive. Then the DAY CAP: if the day's stakes (all rules, both assets, by event date) exceed --day-cap of
   the bankroll they are scaled down together, so a day on which everything loses costs at most that.
4. TEST. Each scheme is run over the whole history with each day's real joint outcomes (so same-day losses across levels,
   rules and assets are captured), on a fixed bankroll (no compounding).
Prices: Polymarket has no prices before 2026, so each trade is priced at the 2026 average for its rule, asset and distance
on days filter v1 keeps. Risk, drawdown and the ranking of schemes are more reliable than the $ returns.
"""
import argparse, os
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default="research/data")
ap.add_argument("--bankroll", type=float, default=20000); ap.add_argument("--day-cap", type=float, default=0.10, help="largest loss in one day as a share of bankroll")
ap.add_argument("--kelly", type=float, default=0.10, help="fraction of the Kelly share to stake"); ap.add_argument("--safety", type=float, default=1.75, help="multiplier on the predicted loss rate")
ap.add_argument("--price-by", choices=["distance", "z"], default="distance", help="price each historical trade at the 2026 average for its distance in %% (default), or for its distance in typical moves (z) — a check that the result does not rest on the market ignoring volatility")
ap.add_argument("--trade-cap", type=float, default=1500, help="largest stake on one trade ($)"); ap.add_argument("--fee-rate", type=float, default=0.01)
a = ap.parse_args()
P = lambda n: os.path.join(a.data_dir, n)
HOURS = {"carry": 24, "sellDn": 6, "sellUp": 9}; Z3 = {"carry": 2.5, "sellDn": 3.5, "sellUp": 2.5}
wavg = lambda v, w: float(np.average(v, weights=w)) if len(v) and np.sum(w) > 0 else np.nan

# ---------- data ----------
R, PRICE, ZPRICE = [], {}, {}
for asset, pre in (("ethereum", "eth"), ("bitcoin", "btc")):
    r = pd.read_pickle(P(f"c12_{pre}_R.pkl")); r["asset"] = asset; R.append(r)
    f = pd.read_pickle(P(f"c12_{pre}_F.pkl"))
    for cell in HOURS:
        y = f[f.cell == cell]; dists = sorted(r[r.cell == cell].dist.unique()); dd = np.array(dists, dtype=float)
        if not len(y): continue
        near = dd[np.abs(y.pct.values[:, None] - dd[None, :]).argmin(axis=1)]; kept = y.keep.values.astype(bool)
        # price by z: the 2026 fills of this rule and asset on v1-kept days, in six equal-sized z groups ($-weighted price per group)
        yk = y[kept & (y.rvv.values > 0)]
        if len(yk) >= 120:
            zf = yk.pct.values / (yk.rvv.values / np.sqrt(365) * np.sqrt(HOURS[cell] / 24)); edges = np.quantile(zf, [1/6, 2/6, 3/6, 4/6, 5/6])
            grp = np.searchsorted(edges, zf); ZPRICE[(asset, cell)] = (edges, [wavg(yk.q.values[grp == g_], yk.usd.values[grp == g_]) for g_ in range(6)])
        for d_ in dists:
            m_ = (near == d_) & kept; m2 = near == d_
            PRICE[(asset, cell, d_)] = wavg(y.q.values[m_], y.usd.values[m_]) if m_.sum() >= 20 else (wavg(y.q.values[m2], y.usd.values[m2]) if m2.sum() >= 20 else np.nan)
R = pd.concat(R); R = R[R.cell.isin(HOURS) & (R.rvv > 0) & R.r24v.notna() & R.trendv.notna()].copy()
R["sigd"] = R.rvv / np.sqrt(365); R["z"] = R.dist / (R.sigd * np.sqrt(R.cell.map(HOURS) / 24)); R["year"] = [d.year for d in R.D]
R["rngz"] = (R.rngv / R.sigd).fillna(0.0); R["r24z"] = R.r24v / R.sigd
R["price"] = [PRICE.get((a_, c_, d_), np.nan) for a_, c_, d_ in zip(R.asset, R.cell, R.dist)]
if a.price_by == "z":      # a trade further away than anything in the 2026 fills gets the price of the furthest group (the smallest premium seen)
    R["price"] = [ZPRICE[(a_, c_)][1][int(np.searchsorted(ZPRICE[(a_, c_)][0], z_))] if (a_, c_) in ZPRICE else np.nan for a_, c_, z_ in zip(R.asset, R.cell, R.z)]
R = R[R.price.notna()].reset_index(drop=True); R["ev"] = R.ev.astype(float); R["v1"] = R.keep.astype(bool)
R["v3"] = R.v1 & (R.z >= R.cell.map(Z3)) & ~((R.cell == "sellDn") & (R.dow == "Sun")) & ~((R.cell == "sellUp") & (R.dow == "Mon"))
print(f"##### sizing study: {R.D.nunique()} days ({R.D.min()} .. {R.D.max()}), {len(R):,} possible selling trades on ETH and BTC; bankroll ${a.bankroll:,.0f}, "
      f"day cap {a.day_cap:.0%} = ${a.bankroll*a.day_cap:,.0f}, Kelly fraction {a.kelly}, safety x{a.safety}, trade cap ${a.trade_cap:,.0f}, prices by {a.price_by}")
if a.price_by == "z":
    for (a_, c_), (ed_, pr_) in ZPRICE.items(): print(f"   2026 price by z, {a_} {c_}: " + ", ".join(f"z<{e_:.1f}: {p_:.4f}" for e_, p_ in zip(ed_, pr_[:-1])) + f", beyond: {pr_[-1]:.4f}")

# ---------- 1. probability model, out of sample by year ----------
def fit(X, y, l2=1.0, it=60):
    b = np.zeros(X.shape[1]); pen = np.r_[0, np.ones(X.shape[1] - 1)]
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ b))); g = X.T @ (y - p) - l2 * pen * b; Hs = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(pen)
        s = np.linalg.solve(Hs, g); b += s
        if np.abs(s).max() < 1e-8: break
    return b
def feats(d, cell):
    cols = [np.ones(len(d)), np.log(d.z.values), np.minimum(d.r24z.values, 0), np.maximum(d.r24z.values, 0), (d.trendv.values >= 0).astype(float), (d.asset.values == "bitcoin").astype(float)]
    names = ["const", "log z", "24h fall", "24h rise", "above 30d avg", "bitcoin"]
    if cell != "carry": cols.append(d.rngz.values); names.append("range so far")
    return np.column_stack(cols), names
R["p"] = np.nan
print("\n################ 1. LOSS-PROBABILITY MODEL (every prediction is for a year the model was not fitted on) ################")
for cell in HOURS:
    idx = R.index[R.cell == cell]; d = R.loc[idx]; X, names = feats(d, cell); y = d.ev.values
    for yr in sorted(d.year.unique()):
        te = (d.year == yr).values; R.loc[idx[te], "p"] = 1 / (1 + np.exp(-(X[te] @ fit(X[~te], y[~te]))))
    d = R.loc[idx]; b = fit(X, y)
    print(f"\n-- {cell}: {len(d):,} trades; effect of each input (log-odds; negative = safer): " + ", ".join(f"{n_} {v_:+.2f}" for n_, v_ in zip(names[1:], b[1:])))
    cal = d.assign(g=pd.qcut(d.p, 5, labels=["safest fifth", "2nd", "3rd", "4th", "riskiest fifth"])).groupby("g", observed=True).agg(trades=("ev", "size"), z=("z", "median"), predicted=("p", "mean"), actual=("ev", "mean"))
    cal[["predicted", "actual"]] = (cal[["predicted", "actual"]] * 100).round(2); cal["z"] = cal.z.round(2); print(cal.to_string())
    yr = d.groupby("year").agg(predicted=("p", "mean"), actual=("ev", "mean")); yr["actual / predicted"] = (yr.actual / yr.predicted).round(2)
    print("   by year, actual / predicted: " + ", ".join(f"{y_}: {v_}" for y_, v_ in yr["actual / predicted"].items()) + f"   (worst {yr['actual / predicted'].max()})")

# ---------- 2. cautious edge and Kelly share ----------
R["be"] = 1 - R.price                                   # loss rate the price can afford
R["edge"] = R.be - R.p                                  # at the predicted rate
R["edge_c"] = R.be - a.safety * R.p                     # cautious
R["kelly"] = (R.edge_c / R.be).clip(lower=0)            # (q - p) / (1 - p) with the cautious q
print("\n################ 2. WHAT THE RULE STAKES, by distance in typical moves (filter v1 trades) ################")
x = R[R.v1].assign(zb=pd.cut(R.z[R.v1], [0, 2, 2.5, 3, 3.5, 4, 5, 99]))
t = x.groupby(["cell", "zb"], observed=True).agg(trades=("ev", "size"), price=("price", "mean"), afford=("be", "mean"), predicted=("p", "mean"), actual=("ev", "mean"), cautious_edge=("edge_c", "mean"),
                                                 staked=("kelly", lambda k: (k > 0).mean()), kelly=("kelly", "mean"))
t["typical stake $"] = (t.kelly * a.kelly * a.bankroll).clip(upper=a.trade_cap).round(0)
for c_ in ("afford", "predicted", "actual", "cautious_edge"): t[c_] = (t[c_] * 100).round(2)
t["staked"] = (t.staked * 100).round(0); t["price"] = t.price.round(4); t = t.drop(columns="kelly").rename(columns={"afford": "price affords %", "predicted": "predicted loss %", "actual": "actual loss %", "cautious_edge": "cautious edge pts", "staked": "% of trades staked"})
print(t.to_string())

# ---------- 3. schemes ----------
CAP = a.bankroll * a.day_cap
def run(mask, raw, name, direction_cap=False):
    x = R[mask].copy(); x["raw"] = np.minimum(np.asarray(raw)[np.asarray(mask)], a.trade_cap); x = x[x.raw > 0]
    key = [x.D, np.where(x.cell == "sellUp", "up", "down")] if direction_cap else [x.D]
    tot = x.groupby(key).raw.transform("sum"); x["stake"] = x.raw * np.minimum(1.0, CAP / tot)
    win = 1 - x.ev; fee = a.fee_rate * (1 - x.price)
    x["pnl"] = x.stake * (win / x.price - 1 - fee); x["pnl1"] = x.stake * (win / np.minimum(x.price + 0.01, 0.999) - 1 - fee)
    d = x.groupby("D").agg(pnl=("pnl", "sum"), pnl1=("pnl1", "sum"), stake=("stake", "sum")).sort_index()
    eq = d.pnl.cumsum(); dd = (eq - eq.cummax()).min(); yrs = (d.index[-1] - d.index[0]).days / 365.25
    yr = d.pnl.groupby(np.array([t_.year for t_ in d.index])).sum(); mon = d.pnl.groupby(np.array([t_.year * 100 + t_.month for t_ in d.index])).sum()
    return dict(scheme=name, trades=len(x), **{"days traded": len(d), "avg staked/day $": round(d.stake.mean()), "total P&L $": round(d.pnl.sum()), "per $ staked %": round(d.pnl.sum() / d.stake.sum() * 100, 2),
                "% of bankroll / yr": round(d.pnl.sum() / a.bankroll / yrs * 100, 1), "loss rate %": round(wavg(x.ev.values, x.stake.values) * 100, 2), "losing days": int((d.pnl < 0).sum()),
                "worst day $": round(d.pnl.min()), "worst year $": round(yr.min()), "losing years": int((yr < 0).sum()), "max drawdown $": round(dd), "drawdown % bankroll": round(dd / a.bankroll * 100, 1),
                "months +": f"{int((mon > 0).sum())}/{len(mon)}", "P&L if prices 1pt worse $": round(d.pnl1.sum())}), d, yr, x
K = a.kelly * a.bankroll * R.kelly.values
tier = np.select([R.z >= R.cell.map(Z3) + 1.5, R.z >= R.cell.map(Z3) + 0.5], [1500.0, 1000.0], 500.0)
SCHEMES = [("A. flat, every v1 trade", R.v1.values, np.full(len(R), 1000.0)),
           ("B. flat, every v3 trade", R.v3.values, np.full(len(R), 1000.0)),
           ("C. v3, three tiers by distance ($500 / $1,000 / $1,500)", R.v3.values, tier),
           (f"D. edge-sized (Kelly x{a.kelly}), v1 trades", R.v1.values, K),
           (f"E. edge-sized (Kelly x{a.kelly}), v3 trades", R.v3.values, K),
           (f"F. edge-sized (Kelly x{a.kelly * 2.5:g}), v1 trades", R.v1.values, K * 2.5),
           (f"G. edge-sized (Kelly x{a.kelly / 2:g}), v1 trades", R.v1.values, K / 2)]
print(f"\n################ 3. SIZING SCHEMES over the whole history (bankroll ${a.bankroll:,.0f} fixed; a day's stakes scaled down to ${CAP:,.0f} if they exceed it) ################")
res = [run(m_, r_, n_) for n_, m_, r_ in SCHEMES]
print(pd.DataFrame([r_[0] for r_ in res]).to_string(index=False))
print("\n-- P&L by year ($) --"); print(pd.DataFrame({r_[0]["scheme"].split(".")[0]: r_[2] for r_ in res}).round(0).fillna(0).to_string())
print("\n-- the same schemes with the cap applied separately to downside (carry + sell downside) and upside (sell upside): each side can lose the cap --")
print(pd.DataFrame([run(m_, r_, n_, True)[0] for n_, m_, r_ in SCHEMES]).to_string(index=False))

# ---------- 4. how losses cluster, and what edge-sizing changed ----------
print("\n################ 4. HOW LOSSES CLUSTER (filter v1 trades, unsized) ################")
x = R[R.v1]; byday = x.groupby("D").agg(trades=("ev", "size"), losses=("ev", "sum")); ld = byday[byday.losses > 0]
g = x[x.ev == 1].assign(side=lambda q: np.where(q.cell == "sellUp", "up", "down")).groupby("D")
print(f"days with trades {len(byday):,}; days with at least one loss {len(ld)} ({len(ld)/len(byday):.1%}); on those days, median share of the day's trades that lost {100*(ld.losses/ld.trades).median():.0f}%; "
      f"days when every trade lost {int((ld.losses == ld.trades).sum())}")
print(f"loss days where both assets lost {int((g.asset.nunique() > 1).sum())} of {g.ngroups}; loss days where both the downside and the upside rules lost {int((g.side.nunique() > 1).sum())}")
dD = res[3][3]; share = dD.groupby("cell").stake.sum() / dD.stake.sum(); zmix = dD.assign(zb=pd.cut(dD.z, [0, 2.5, 3.5, 5, 99])).groupby("zb", observed=True).stake.sum() / dD.stake.sum()
dA = res[0][3]; shareA = dA.groupby("cell").stake.sum() / dA.stake.sum(); zmixA = dA.assign(zb=pd.cut(dA.z, [0, 2.5, 3.5, 5, 99])).groupby("zb", observed=True).stake.sum() / dA.stake.sum()
print("where the money goes, flat (A) vs edge-sized (D) — by rule: " + ", ".join(f"{c_} {shareA.get(c_, 0):.0%} -> {share.get(c_, 0):.0%}" for c_ in HOURS)
      + ";  by distance in typical moves: " + ", ".join(f"{k_} {zmixA.get(k_, 0):.0%} -> {zmix.get(k_, 0):.0%}" for k_ in zmix.index))
ass = dD.groupby("asset").stake.sum() / dD.stake.sum(); print("edge-sized (D) by asset: " + ", ".join(f"{k_} {v_:.0%}" for k_, v_ in ass.items()))
