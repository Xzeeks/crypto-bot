#!/usr/bin/env python3
"""c2_carry_grid.py — the "X% away, H hours out, priced at Y" sweet-spot search.

    python research/scripts/c2_carry_grid.py research/data/eth_above.duckdb
    python research/scripts/c2_carry_grid.py research/data/btc_above.duckdb --step 2000

Table 1 (klines only): for every event-day, every lookback H hours before the noon print and every strike on the
  grid (default $100 step) within 15% of spot: did the favoured side (spot above strike -> Yes, below -> No)
  survive to the print?  -> empirical loss rate per (|pct away| x hours-out) cell.  This is the TRUE risk.
Table 2 (taker fills): every fill restated as "buy the favoured side at price q", bucketed the same way:
  n, $ notional, average q, best (lowest) q seen, realized win rate, losses, gross ROI per $1.
Table 3: sweet-spot candidates = cells where Table 1 loss rate <= --max-loss AND fills exist with q <= --max-q.
"""
import argparse, math
import duckdb, numpy as np, pandas as pd
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("--step", type=float, default=100.0, help="strike grid step ($)")
ap.add_argument("--max-loss", type=float, default=0.005, help="max empirical loss rate for a sweet spot")
ap.add_argument("--max-q", type=float, default=0.97, help="max favoured-side price for a sweet spot")
a = ap.parse_args()
con = duckdb.connect(a.db, read_only=True)

PCT = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10, 100]
HRS = [0, 0.25, 0.5, 1, 2, 3, 6, 9, 12, 18, 24, 48, 96, 1e9]
HRS_LBL = ["<15m", "15-30m", "30m-1h", "1-2h", "2-3h", "3-6h", "6-9h", "9-12h", "12-18h", "18-24h", "1-2d", "2-4d", ">4d"]

# ---------- Table 1: empirical survival from klines ----------
kl = con.execute("SELECT open_ts, close FROM klines ORDER BY open_ts").df().set_index("open_ts")["close"]
days = con.execute("SELECT DISTINCT resolve_ts FROM markets WHERE outcome IS NOT NULL ORDER BY 1").df().resolve_ts.tolist()
rows = []
lookbacks_h = [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 36, 48, 72, 96]
for rts in days:
    if (rts - 60) not in kl.index: continue
    final = kl.loc[rts - 60]                      # the resolving candle (11:59 open, closes at 12:00)
    for h in lookbacks_h:
        t = rts - int(h * 3600); t -= t % 60
        if t not in kl.index: continue
        spot = kl.loc[t]
        lo = math.floor(spot * 0.85 / a.step) * a.step
        hi = math.ceil(spot * 1.15 / a.step) * a.step
        k = lo
        while k <= hi:
            if k > 0 and abs(spot - k) > 1e-9:
                fav_yes = spot > k
                won = (final > k) == fav_yes
                rows.append((h, abs(spot - k) / k * 100, won))
            k += a.step
t1 = pd.DataFrame(rows, columns=["hours", "pct", "won"])
t1["pct_b"] = pd.cut(t1.pct, PCT); t1["h_b"] = pd.cut(t1.hours, HRS, labels=HRS_LBL)
g1 = t1.groupby(["h_b", "pct_b"], observed=True).agg(n=("won", "size"), losses=("won", lambda s: int((~s).sum())))
g1["loss_rate"] = g1.losses / g1.n
print(f"=== Table 1: empirical loss rate of the favoured side (klines, {len(days)} days, ${a.step:.0f} grid) ===")
piv = g1.loss_rate.unstack("pct_b")
print((piv * 100).round(2).to_string())
print("\nn per cell:"); print(g1.n.unstack("pct_b").to_string())

# ---------- Table 2: what the market charged ----------
f = con.execute("""
    SELECT f.ts, f.price, f.size, f.token, f.side, m.strike, m.resolve_ts, m.outcome, m.yes_token
    FROM fills f JOIN markets m USING (condition_id) WHERE m.outcome IS NOT NULL AND f.ts < m.resolve_ts""").df()
f["minute"] = (f.ts // 60) * 60
f["spot"] = f.minute.map(kl)
f = f.dropna(subset=["spot"])
f["hours"] = (f.resolve_ts - f.ts) / 3600
f["pct"] = (f.spot - f.strike).abs() / f.strike * 100
f["fav_yes"] = f.spot > f.strike
f["p_yes"] = np.where(f.token == f.yes_token, f.price, 1 - f.price)
f["q"] = np.where(f.fav_yes, f.p_yes, 1 - f.p_yes)               # price of the favoured side
f["fav_won"] = np.where(f.fav_yes, f.outcome == 1, f.outcome == 0)
f["usd"] = f["size"] * f["q"]              # favoured-side notional (1,000 No @0.02 == 1,000 Yes @0.98 == $980 deployed)
f["pnl"] = f["size"] * (f["fav_won"].astype(float) - f["q"])   # P&L of buying the favoured side on this fill
f["pct_b"] = pd.cut(f.pct, PCT); f["h_b"] = pd.cut(f.hours, HRS, labels=HRS_LBL)
f["q_b"] = pd.cut(f.q, [0, .5, .8, .9, .95, .97, .98, .99, .995, 1.0001])
g2 = f.groupby(["h_b", "pct_b"], observed=True).agg(
    n=("q", "size"), usd=("usd", "sum"), avg_q=("q", "mean"), min_q=("q", "min"), p10_q=("q", lambda s: s.quantile(.1)),
    win=("fav_won", "mean"), losses=("fav_won", lambda s: int((~s).sum()))).reset_index()
g2["roi"] = (g2.win - g2.avg_q) / g2.avg_q
g2 = g2.merge(g1.reset_index()[["h_b", "pct_b", "loss_rate"]], on=["h_b", "pct_b"], how="left").rename(columns={"loss_rate": "true_loss"})
g2["usd"] = g2.usd.round(0)
print("\n=== Table 2: favoured-side fills by cell (avg_q = what you paid on average, min_q = best price seen) ===")
print(g2.round(4).to_string(index=False))

# ---------- Table 3: candidates ----------
cand = g2[(g2.true_loss <= a.max_loss) & (g2.p10_q <= a.max_q) & (g2.n >= 30)].copy()
print(f"\n=== Table 3: candidates — true loss <= {a.max_loss:.1%}, 10th-pct price <= {a.max_q}, n >= 30 ===")
if cand.empty:
    print("none")
else:
    print(cand.round(4).to_string(index=False))

# ---------- Table 4: every near-certain cell as a daily business ----------
print(f"\n=== Table 4: cells with true loss <= {a.max_loss:.1%} and fill ROI > 0.3%: capacity and loss concentration ===")
biz = g2[(g2.true_loss <= a.max_loss) & (g2.roi > 0.003) & (g2.n >= 100)][["h_b", "pct_b"]]
out = []
for _, r in biz.iterrows():
    s = f[(f.h_b == r.h_b) & (f.pct_b == r.pct_b)]
    byday = s.groupby("resolve_ts").agg(usd=("usd", "sum"), pnl=("pnl", "sum"), lossn=("fav_won", lambda w: int((~w).sum())))
    byday = byday.reindex(days, fill_value=0)
    out.append(dict(h_b=r.h_b, pct_b=r.pct_b, fills=len(s), usd_total=s.usd.sum(), usd_per_day=s.usd.sum() / len(days),
                    median_day=byday.usd.median(), days_with_flow=int((byday.usd > 0).sum()), avg_q=s.q.mean(),
                    loss_days=int((byday.lossn > 0).sum()), worst_day_pnl=byday.pnl.min(), gross_pnl=byday.pnl.sum(),
                    roi=byday.pnl.sum() / max(s.usd.sum(), 1e-9)))
t4 = pd.DataFrame(out).sort_values("gross_pnl", ascending=False)
if t4.empty:
    print("none")
else:
    for c in ("usd_total", "usd_per_day", "median_day", "worst_day_pnl", "gross_pnl"): t4[c] = t4[c].round(0)
    print(t4.round(4).to_string(index=False))
    print("\n(gross_pnl = P&L if you had been the taker on EVERY fill in the cell at its price; usd_per_day = the ceiling of what "
          "could be deployed; loss_days = event-days containing at least one losing fill)")

# ---------- the best price ever offered in each near-certain cell, with $ at that price ----------
print(f"\n=== $ available at q <= {a.max_q} in cells with true loss <= {a.max_loss:.1%} (capacity check) ===")
safe = g1[g1.loss_rate <= a.max_loss].reset_index()[["h_b", "pct_b"]]
cheap = f.merge(safe, on=["h_b", "pct_b"])
cheap = cheap[cheap.q <= a.max_q]
if cheap.empty:
    print("none")
else:
    per_day = cheap.groupby(cheap.resolve_ts).usd.sum()
    print(f"fills={len(cheap)} usd=${cheap.usd.sum():,.0f} over {len(days)} days -> ${cheap.usd.sum()/len(days):,.0f}/day "
          f"(median day ${per_day.median():,.0f}); realized win {cheap.fav_won.mean():.4f}; avg q {cheap.q.mean():.3f}; "
          f"gross ROI {(cheap.fav_won.mean()-cheap.q.mean())/cheap.q.mean():+.2%}")
    print(cheap.groupby(["h_b", "q_b"], observed=True).agg(n=("q", "size"), usd=("usd", "sum"), win=("fav_won", "mean")).round(3).to_string())
