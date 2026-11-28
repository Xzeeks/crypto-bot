#!/usr/bin/env python3
"""c17_expected_value.py — are today's prices paying more than the long-run risk? Expected value of every shadow trade.

    python research/scripts/c11_shadow_review.py research/data/books.sqlite > research/data/c11_review.txt      # writes c11_trades.pkl
    python research/scripts/c17_expected_value.py [--data-dir research/data] [--fee-rate 0.01]

PRICE comes from the shadow trades (c11_trades.pkl: what a taker could actually have bought at, from the logged books).
RISK comes from the long price history (c12_<a>_R.pkl: one simulated trade per day per distance over ~7 years).
For each shadow trade the comparable historical trades are: same rule and asset; a similar distance measured in typical
moves (z, using the hours the trade actually had left); the same 24h-move bucket; the same side of the 30-day average.
If fewer than --min-n comparable trades exist the match is loosened step by step (wider z band, then drop the trend
condition, then drop the 24h-move condition) and the step used is reported.
  break-even%   the loss rate the price can afford (1 - price); for buyT the hit rate it needs (price)
  hist%         loss rate (hit rate for buyT) of the comparable historical trades, all years pooled
  worst-yr%     the worst single calendar year among them (years with at least 30 comparable trades)
  EV/$          expected profit per $ staked after the taker fee, using hist% / using worst-yr%
Nothing here is tuned and no rule changes: it is a readout of current prices against long-run frequencies.
Known bias: the shadow review enters when the price first drops under its limit, i.e. when the market has just become
more worried; the historical match cannot see that, so true risk on those entries may be somewhat higher than hist%.
"""
import argparse, os
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default="research/data"); ap.add_argument("--fee-rate", type=float, default=0.01)
ap.add_argument("--min-n", type=int, default=150)
a = ap.parse_args()
P = lambda n: os.path.join(a.data_dir, n)
T = pd.read_pickle(P("c11_trades.pkl"))
HOURS = {"carry": 24, "sellDn": 6, "sellUp": 9, "buyT": 18}; SELL = {"carry": True, "sellDn": True, "sellUp": True, "buyT": False}
def bucket(r):
    if r is None or r != r: return None
    return "fell >4%" if r <= -4 else ("fell 2-4%" if r <= -2 else ("flat" if r < 2 else ("rose 2-4%" if r < 4 else "rose >4%")))
HIST = {}
for asset, pre in (("ethereum", "eth"), ("bitcoin", "btc")):
    R = pd.read_pickle(P(f"c12_{pre}_R.pkl")); R = R[R.rvv > 0].copy()
    R["z"] = R.dist.astype(float) / (R.rvv.astype(float) / np.sqrt(365) * np.sqrt(R.cell.map(HOURS).astype(float) / 24))
    R["r24b"] = [bucket(x) for x in R.r24v]; R["above"] = R.trendv >= 0; R["year"] = [d.year for d in R.D]
    for code in HOURS: HIST[(asset, code)] = R[R.cell == code][["z", "r24b", "above", "year", "ev", "D"]].reset_index(drop=True)
print(f"##### expected value of the shadow trades: {T.event.nunique()} event-days ({T.event.min()} .. {T.event.max()}); history {min(h.D.min() for h in HIST.values())} .. {max(h.D.max() for h in HIST.values())}; taker fee rate {a.fee_rate}")
STEPS = [(0.15, True, True, "z ±15%, same 24h move, same trend side"), (0.25, True, True, "z ±25%, same 24h move, same trend side"),
         (0.25, False, True, "z ±25%, same 24h move"), (0.25, False, False, "z ±25% only"), (0.40, False, False, "z ±40% only")]
def lookup(asset, code, z, r24, trend):
    X = HIST[(asset, code)]; b = bucket(r24); ab = None if trend is None else (trend == "above")
    for i, (band, use_tr, use_r, _) in enumerate(STEPS):
        m = X.z.between(z * (1 - band), z * (1 + band)).values.copy()
        if use_tr and ab is not None: m &= (X.above.values == ab)
        if use_r and b is not None: m &= (X.r24b.values == b)
        if m.sum() >= a.min_n or i == len(STEPS) - 1: break
    sub = X[m]
    if not len(sub): return np.nan, np.nan, 0, i
    yr = sub.groupby("year").ev.agg(["mean", "size"]); yr = yr[yr["size"] >= 30]["mean"]
    worst = (yr.max() if SELL[code] else yr.min()) if len(yr) else np.nan
    return float(sub.ev.mean()), float(worst), int(m.sum()), i
rows = []
for r in T.itertuples():
    rv = r.rv if (r.rv is not None and r.rv == r.rv and r.rv > 0) else np.nan
    z = r.pct / (rv / np.sqrt(365) * np.sqrt(max(r.hrs, 0.25) / 24)) if rv == rv else np.nan      # distance in typical moves over the time actually left
    rate, worst, n, step = lookup(r.asset, r.code, z, r.r24, r.trend) if z == z else (np.nan, np.nan, 0, len(STEPS) - 1)
    p = r.vwap; fee = a.fee_rate * (1 - p)                                                           # fee per $ staked
    if SELL[r.code]: be = 1 - p; ev = (1 - rate) / p - 1 - fee; evw = (1 - worst) / p - 1 - fee
    else: be = p; ev = rate / p - 1 - fee; evw = worst / p - 1 - fee
    rows.append(dict(strategy=r.strategy, cell=r.cell, code=r.code, event=r.event, market=r.market, entry=r.entry, hrs=r.hrs, pct=r.pct, z_left=round(z, 2) if z == z else np.nan, r24=r.r24, trend=r.trend,
                     price=p, usd=r.usd, be=be, hist=rate, worst=worst, n=n, step=step, ev=ev, evw=evw, paper=np.nan if r.pnl is None or r.pnl != r.pnl else r.pnl - r.fee))
E = pd.DataFrame(rows); E.to_pickle(P("c17_ev.pkl"))
w = lambda x, col: float(np.average(x[col], weights=x.usd)) if len(x) and x[col].notna().all() else (float(np.average(x[col][x[col].notna()], weights=x.usd[x[col].notna()])) if x[col].notna().any() else np.nan)
def summarise(x, lab):
    sell = SELL[x.code.iloc[0]] if x.code.nunique() == 1 else True
    return {"": lab, "trades": len(x), "staked $": round(x.usd.sum()), "avg price": round(w(x, "price"), 4), "break-even%": round(w(x, "be") * 100, 2), "hist%": round(w(x, "hist") * 100, 2),
            "worst-yr%": round(w(x, "worst") * 100, 2), "margin pts": round((w(x, "be") - w(x, "hist")) * 100 * (1 if sell else -1), 2), "EV/$ %": round(w(x, "ev") * 100, 2), "EV/$ worst-yr %": round(w(x, "evw") * 100, 2),
            "expected $": round((x.usd * x.ev).sum()), "expected $ worst-yr": round((x.usd * x.evw).sum()), "paper $": round(x.paper.sum()), "comparable n (median)": int(x.n.median())}
order = [c for c in ["ETH carry", "BTC carry", "ETH sellDn", "BTC sellDn", "ETH sellUp", "BTC sellUp", "ETH buyT", "BTC buyT"] if c in set(E.cell)]
for s_ in ("none", "v1", "v2", "v3"):
    x = E[E.strategy == s_]
    if x.empty: continue
    out = [summarise(x[x.cell == c], c) for c in order if (x.cell == c).any()]
    sb = x[x.code != "buyT"]
    if len(sb): out.append(summarise(sb, "SELLING BOOK"))
    print(f"\n================ strategy: {'no filter' if s_ == 'none' else s_} ================")
    print("(for buyT: break-even% is the hit rate the price needs, hist% the hit rate history gives, worst-yr% the lowest year; margin = hist - break-even)")
    print(pd.DataFrame(out).to_string(index=False))
print("\n-- how the historical match was made (share of trades by step) --")
print((E.groupby("code").step.value_counts(normalize=True).unstack(fill_value=0).rename(columns={i: STEPS[i][3] for i in range(len(STEPS))}) * 100).round(0).to_string())
print("\n-- selling trades, no filter, by distance in typical moves over the time left (is the market paying more than history at each distance?) --")
sb = E[(E.strategy == "none") & (E.code != "buyT")].copy(); sb["zb"] = pd.cut(sb.z_left, [0, 1.5, 2, 2.5, 3, 4, 99])
print(pd.DataFrame([dict({"z": str(k), "rule": c}, **{kk: vv for kk, vv in summarise(g, "").items() if kk in ("trades", "staked $", "break-even%", "hist%", "worst-yr%", "margin pts", "EV/$ %", "EV/$ worst-yr %")})
                    for (c, k), g in sb.groupby(["code", "zb"], observed=True)]).to_string(index=False))
