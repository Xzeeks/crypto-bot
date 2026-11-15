#!/usr/bin/env python3
"""
c1_eth_above_analyze.py — first-cut read of the "Ethereum above $X on <date>" ladder from the c1 pull.

    python c1_eth_above_analyze.py research/data/eth_above.duckdb

Every taker fill is re-expressed as a BUY of the Yes contract at price p_yes (a No buy at q == Yes sell at 1-q),
tagged with spot at fill time (Binance 1m close), time-to-expiry tau, realized vol sigma (trailing 24h of 1m
log returns, scaled), moneyness z = ln(S/K) / (sigma*sqrt(tau)) and model fair = N(d2).
Outcome = market outcome (Yes=1). P&L of buying Yes at p: outcome - p; of buying No at q: (1-outcome) - q.

Reports
  A. Calibration grid: tau bucket x price bucket -> n, $ notional, realized hit rate, avg price, avg model fair
  B. Longshot skew: far-OTM (z <= -2) and far-ITM (z >= 2) fills at >= 24h: market price vs fair vs realized
  C. Near-expiry carry: fills in the last 3h priced 0.90-0.995, by |z| bucket: loss rate, ROI, $ capacity/day
  D. Post-print fills: dollars traded after the 12:00 ET candle closed (12:01 ET), by price
  E. Ladder monotonicity: same-event, same-minute Yes mids where P(above K1) < P(above K2) with K1 < K2
"""
import math, sys
import duckdb, numpy as np, pandas as pd

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

def ncdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))

db = sys.argv[1]
con = duckdb.connect(db, read_only=True)

kl = con.execute("SELECT open_ts, close FROM klines ORDER BY open_ts").df()
kl["lr"] = np.log(kl["close"]).diff()
# trailing 24h realized vol per minute, annualized to per-second units via sqrt(seconds)
kl["sig_1m"] = kl["lr"].rolling(1440, min_periods=300).std()
kl["sig_per_sqrt_s"] = kl["sig_1m"] / math.sqrt(60)
kl = kl.dropna(subset=["sig_per_sqrt_s"])

f = con.execute("""
    SELECT f.*, m.event_date, m.strike, m.resolve_ts, m.outcome, m.yes_token
    FROM fills f JOIN markets m USING (condition_id)
    WHERE m.outcome IS NOT NULL""").df()
if f.empty:
    sys.exit("no resolved fills")

f["is_yes"] = f["token"] == f["yes_token"]
# normalise to a Yes-buy price; p&l for the taker on this fill
f["p_yes"] = np.where(f["is_yes"], f["price"], 1 - f["price"])
f["taker_buys_yes"] = np.where(f["is_yes"], f["side"] == "BUY", f["side"] == "SELL")
f["taker_pnl_per_share"] = np.where(f["taker_buys_yes"], f["outcome"] - f["p_yes"], (1 - f["outcome"]) - (1 - f["p_yes"]))
f["notional"] = f["size"] * f["price"]
f["tau"] = f["resolve_ts"] - f["ts"]          # seconds until the 12:00 candle CLOSES
f["minute_ts"] = (f["ts"] // 60) * 60
f = f.merge(kl[["open_ts", "close", "sig_per_sqrt_s"]], left_on="minute_ts", right_on="open_ts", how="left")
f = f.dropna(subset=["close"])
tau_pos = f["tau"].clip(lower=1)
sig_t = f["sig_per_sqrt_s"] * np.sqrt(tau_pos)
f["z"] = np.log(f["close"] / f["strike"]) / sig_t
f["fair"] = [ncdf(z - 0.5 * s) for z, s in zip(f["z"], sig_t)]   # N(d2)
f["edge_vs_fair"] = f["p_yes"] - f["fair"]

print(f"\nfills={len(f):,}  events={f.event_date.nunique()}  strikes={f.condition_id.nunique()}  "
      f"notional=${f.notional.sum():,.0f}  resolution window {f.event_date.min()} .. {f.event_date.max()}")

tau_bins = [-1e9, 0, 600, 3600, 3*3600, 24*3600, 3*86400, 1e12]
tau_lbl = ["post-print", "<10m", "10m-1h", "1-3h", "3-24h", "1-3d", ">3d"]
p_bins = [0, .05, .10, .20, .35, .50, .65, .80, .90, .95, .98, 1.0001]
f["tau_b"] = pd.cut(f["tau"], tau_bins, labels=tau_lbl)
f["p_b"] = pd.cut(f["p_yes"], p_bins)

print("\n=== A. calibration grid (Yes-buy price vs realized Yes rate vs N(d2)) ===")
g = f.groupby(["tau_b", "p_b"], observed=True).agg(
    n=("p_yes", "size"), usd=("notional", "sum"), price=("p_yes", "mean"),
    realized=("outcome", "mean"), fair=("fair", "mean")).reset_index()
g["mkt-real"] = (g["price"] - g["realized"]).round(3)
g["usd"] = g["usd"].round(0)
print(g.to_string(index=False))

print("\n=== B. longshot skew (>=24h to expiry) ===")
far = f[f["tau"] >= 86400]
for lbl, mask in [("far OTM Yes (z<=-2)", far["z"] <= -2), ("far ITM Yes (z>=2)", far["z"] >= 2),
                  ("|z|<0.5 (ATM)", far["z"].abs() < 0.5)]:
    s = far[mask]
    if len(s):
        print(f"{lbl:22s} n={len(s):6d} usd=${s.notional.sum():10,.0f} price={s.p_yes.mean():.3f} "
              f"fair={s.fair.mean():.3f} realized={s.outcome.mean():.3f}")

print("\n=== C. near-expiry carry: last 3h, Yes-equivalent price 0.90-0.995 (buy the likely side) ===")
# treat each fill as 'buy the side priced >= 0.90': favoured side price q = max(p_yes, 1-p_yes)
c = f[(f["tau"] > 0) & (f["tau"] <= 3 * 3600)].copy()
c["q"] = np.maximum(c["p_yes"], 1 - c["p_yes"])
c["fav_wins"] = np.where(c["p_yes"] >= 0.5, c["outcome"], 1 - c["outcome"])
c = c[(c["q"] >= 0.90) & (c["q"] <= 0.995)]
c["absz"] = pd.cut(c["z"].abs(), [0, 1, 1.5, 2, 2.5, 3, 4, 99])
c["q_b"] = pd.cut(c["q"], [.90, .95, .97, .98, .99, .995])
cc = c.groupby(["absz", "q_b"], observed=True).agg(
    n=("q", "size"), usd=("notional", "sum"), price=("q", "mean"), win=("fav_wins", "mean")).reset_index()
cc["roi_per_trade"] = (cc["win"] - cc["price"]) / cc["price"]
cc["usd"] = cc["usd"].round(0)
print(cc.to_string(index=False))
days = f["event_date"].nunique()
print(f"\n$ traded in the clean cell (|z|>=2.5, q 0.95-0.99) per event-day: "
      f"${c[(c.z.abs()>=2.5)&(c.q>=.95)&(c.q<.99)].notional.sum()/max(days,1):,.0f}")

print("\n=== D. post-print fills (after the 12:00 ET candle closed) ===")
pp = f[f["tau"] <= 0]
if len(pp):
    pp2 = pp.assign(correct=(pp["taker_buys_yes"] == (pp["outcome"] == 1)))
    print(pp2.groupby(pd.cut(pp2["price"], [0, .5, .9, .95, .99, 1.0001]), observed=True)
          .agg(n=("price", "size"), usd=("notional", "sum"), taker_correct=("correct", "mean")).to_string())
else:
    print("none — market apparently stops trading at the print")

print("\n=== E. ladder monotonicity violations (same event, same minute, Yes price rises with strike) ===")
m = (f.groupby(["event_date", "minute_ts", "strike"])["p_yes"].mean().reset_index()
       .sort_values(["event_date", "minute_ts", "strike"]))
m["p_next"] = m.groupby(["event_date", "minute_ts"])["p_yes"].shift(-1)
viol = m[(m["p_next"].notna()) & (m["p_next"] > m["p_yes"] + 0.02)]
print(f"minute-pairs={m.p_next.notna().sum():,}  violations(>2c)={len(viol):,}")
print(viol.head(15).to_string(index=False))
