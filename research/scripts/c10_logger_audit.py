#!/usr/bin/env python3
"""c10_logger_audit.py — is the droplet book logger's data complete, accurate and usable?

    python research/scripts/c10_logger_audit.py research/data/books.sqlite --asset ethereum \
        --hit research/data/eth_hit.duckdb --above research/data/eth_above.duckdb

Checks, each against an independent source where one exists:
  A. Continuity   tick spacing (expected 120s), gaps > 5 min, uptime.
  B. Coverage     per event-day: outcomes logged vs markets Polymarket listed (fills DBs); books per tick.
  C. Spot         logged Binance spot vs the 1m kline for that minute (kline low/high and close).
  D. Book sanity  empty sides, crossed books, prices outside (0,1).
  E. Outcomes     touch ladder: logger's early-resolve mark vs resolved outcome. close ladder: provisional vs resolved.
  F. Books vs prints   every taker print vs the logged best bid/ask at the last snapshot before it (<= 150s earlier).
Ends with a PASS / CHECK line per test.
"""
import argparse, sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import duckdb, numpy as np, pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
NY = ZoneInfo("America/New_York")
ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("--asset", default="ethereum")
ap.add_argument("--hit", default=None); ap.add_argument("--above", default=None)
a = ap.parse_args()
verdict = []
def note(name, ok, msg): verdict.append((name, "PASS " if ok else "CHECK", msg))
fmt = lambda t: datetime.fromtimestamp(int(t), NY).strftime("%m-%d %H:%M")

con = sqlite3.connect(a.db)
b = pd.read_sql("SELECT ts, family, event_date, direction, strike, resolve_ts, spot, best_bid, best_ask FROM books WHERE asset=?", con, params=(a.asset,))
sp = pd.read_sql("SELECT ts, price FROM spot WHERE asset=? ORDER BY ts", con, params=(a.asset,))
mk = pd.read_sql("SELECT * FROM markets WHERE asset=?", con, params=(a.asset,))
if b.empty: raise SystemExit("no rows for " + a.asset)
print(f"##### {a.asset}: {len(b):,} book rows, {fmt(b.ts.min())} .. {fmt(b.ts.max())} NY")

# ---------- A. continuity ----------
ticks = np.sort(b.ts.unique()); gaps = np.diff(ticks)
span = ticks[-1] - ticks[0]; lost = gaps[gaps > 300].sum() - 120 * (gaps > 300).sum()
print(f"\n=== A. continuity ===\nticks {len(ticks):,} (expected ~{span // 120 + 1:,}); interval median {np.median(gaps):.0f}s, p99 {np.percentile(gaps, 99):.0f}s, max {gaps.max():.0f}s")
big = [(ticks[i], gaps[i]) for i in np.argsort(-gaps)[:8] if gaps[i] > 300]
for t, g in big: print(f"  gap {g/60:6.1f} min starting {fmt(t)} NY")
up = 1 - lost / span
print(f"gaps > 5 min: {(gaps > 300).sum()}; time not covered: {lost/3600:.1f}h of {span/3600:.0f}h -> uptime {up:.2%}")
print(f"spot rows {len(sp):,}; ticks with no spot: {len(set(ticks) - set(sp.ts))}")
note("A continuity", up > 0.99 and (gaps > 900).sum() == 0, f"uptime {up:.2%}, {(gaps > 300).sum()} gaps > 5 min, longest {gaps.max()/60:.0f} min")

# ---------- B. coverage ----------
print("\n=== B. coverage: outcomes logged per event-day vs markets listed by Polymarket ===")
logged = b.groupby(["family", "event_date"]).apply(lambda g: g[["direction", "strike"]].drop_duplicates().shape[0]).rename("logged").reset_index()
listed = []
for fam, path in (("hit", a.hit), ("above", a.above)):
    if not path: continue
    d = duckdb.connect(path, read_only=True)
    x = d.execute("SELECT CAST(event_date AS VARCHAR) AS event_date, COUNT(*) AS listed FROM markets GROUP BY 1").df(); x["family"] = fam; listed.append(x); d.close()
cov = logged.merge(pd.concat(listed) if listed else pd.DataFrame(columns=["family", "event_date", "listed"]), on=["family", "event_date"], how="left")
cov["first_seen"] = cov.apply(lambda r: fmt(b[(b.family == r.family) & (b.event_date == r.event_date)].ts.min()), axis=1)
print(cov.pivot(index="event_date", columns="family", values=["logged", "listed", "first_seen"]).to_string())
short = cov[cov.listed.notna() & (cov.logged < cov.listed)]
per_tick = b.groupby(["ts", "family"]).size().unstack(fill_value=0)
print(f"books per tick: " + ", ".join(f"{c} median {int(per_tick[c].median())} min {int(per_tick[c].min())}" for c in per_tick.columns))
note("B coverage", short.empty, "all listed outcomes logged" if short.empty else f"{len(short)} event-days with fewer outcomes logged than listed (touch barriers hit before the first poll are expected)")

# ---------- C. spot ----------
kpath = a.hit or a.above
if kpath:
    d = duckdb.connect(kpath, read_only=True); kl = d.execute("SELECT open_ts, low, high, close FROM klines").df().set_index("open_ts"); d.close()
    s = sp.copy(); s["m"] = (s.ts // 60) * 60
    s = s.join(kl, on="m").dropna(subset=["close"])
    nxt = kl.reindex(s.m + 60); prv = kl.reindex(s.m - 60)
    lo = np.fmin(np.fmin(s.low.values, nxt.low.values), prv.low.values); hi = np.fmax(np.fmax(s.high.values, nxt.high.values), prv.high.values)
    inside = ((s.price.values >= lo * (1 - 1e-4)) & (s.price.values <= hi * (1 + 1e-4))).mean()
    dev = (s.price / s.close - 1).abs() * 100
    print(f"\n=== C. spot vs Binance 1m klines ({len(s):,} ticks with a kline) ===\ninside the minute's low-high range (±1 min): {inside:.2%}; "
          f"|spot/close - 1|: median {dev.median():.3f}%, p99 {dev.quantile(.99):.3f}%, max {dev.max():.3f}%")
    note("C spot", inside > 0.995 and dev.quantile(.99) < 0.5, f"{inside:.2%} inside kline range, p99 deviation {dev.quantile(.99):.2f}%")

# ---------- D. book sanity ----------
print("\n=== D. book sanity ===")
for fam, g in b.groupby("family"):
    crossed = (g.best_bid >= g.best_ask).mean(); bad = (((g.best_bid <= 0) | (g.best_bid >= 1)).sum() + ((g.best_ask <= 0) | (g.best_ask >= 1)).sum())
    print(f"{fam:5s} rows {len(g):,}: no bid {g.best_bid.isna().mean():.1%}, no ask {g.best_ask.isna().mean():.1%}, both empty {(g.best_bid.isna() & g.best_ask.isna()).mean():.1%}, "
          f"crossed/locked {crossed:.2%}, prices outside (0,1): {bad}")
crossed_all = (b.best_bid >= b.best_ask).mean()
note("D book sanity", crossed_all < 0.005, f"crossed/locked books {crossed_all:.2%} (one-sided books are normal on far strikes)")

# ---------- E. outcomes ----------
print("\n=== E. outcomes ===")
def wend(ed):
    d0 = datetime.fromisoformat(ed); return int((datetime(d0.year, d0.month, d0.day, tzinfo=NY) + timedelta(days=1)).timestamp())
if a.hit:
    d = duckdb.connect(a.hit, read_only=True)
    o = d.execute("SELECT CAST(event_date AS VARCHAR) AS event_date, direction, strike, outcome FROM markets WHERE outcome IS NOT NULL").df(); d.close()
    h = mk[mk.family == "hit"].copy(); h["window_end"] = h.event_date.map(wend)
    h["mark_touched"] = (h.resolve_ts < h.window_end - 3600).astype(int)
    j = h.merge(o, on=["event_date", "direction", "strike"])
    print("touch ladder — logger's early-resolve mark (rows) vs resolved outcome (cols: 0 = not touched, 1 = touched):")
    print(pd.crosstab(j.mark_touched, j.outcome).to_string())
    wrong = j[j.mark_touched != j.outcome]
    if len(wrong): print("disagreements:"); print(wrong[["event_date", "direction", "strike", "outcome", "mark_touched", "resolve_ts", "last_seen"]].head(12).to_string(index=False))
    note("E touch outcomes", len(wrong) == 0, f"{len(j) - len(wrong)}/{len(j)} marks agree" + ("" if len(wrong) == 0 else " — evaluators must keep using the fills DB outcome, not the mark"))
if a.above:
    d = duckdb.connect(a.above, read_only=True)
    o = d.execute("SELECT CAST(event_date AS VARCHAR) AS event_date, strike, outcome FROM markets WHERE outcome IS NOT NULL").df(); d.close()
    ab = b[(b.family == "above") & (b.ts <= b.resolve_ts)]
    last = ab.loc[ab.groupby(["event_date", "strike"]).ts.idxmax()]
    last = last[last.resolve_ts - last.ts < 900].assign(prov=lambda x: (x.spot > x.strike).astype(int))
    j = last.merge(o, on=["event_date", "strike"]); wrong = j[j.prov != j.outcome]
    print(f"close ladder — provisional outcome from logged spot at the print vs resolved: {len(j) - len(wrong)}/{len(j)} agree")
    if len(wrong): print(wrong[["event_date", "strike", "spot", "outcome"]].to_string(index=False))
    note("E close outcomes", len(wrong) <= 1, f"{len(j) - len(wrong)}/{len(j)} agree (a strike within a few $ of spot at noon can legitimately differ)")

# ---------- F. books vs prints ----------
print("\n=== F. taker prints vs the logged book just before them ===")
for fam, path in (("hit", a.hit), ("above", a.above)):
    if not path: continue
    d = duckdb.connect(path, read_only=True)
    dircol = "m.direction" if fam == "hit" else "''"
    f = d.execute(f"""SELECT CAST(m.event_date AS VARCHAR) AS event_date, {dircol} AS direction, m.strike, f.ts, f.price, f.size, f.side, (f.token = m.yes_token) AS is_yes
                      FROM fills f JOIN markets m USING (condition_id) WHERE f.ts >= {int(b.ts.min())} AND f.ts <= {int(b.ts.max())}""").df(); d.close()
    if f.empty: print(f"{fam}: no prints in the logged period"); continue
    f["p_yes"] = np.where(f.is_yes, f.price, 1 - f.price); f["buy_yes"] = np.where(f.is_yes, f.side == "BUY", f.side == "SELL")
    g = b[b.family == fam][["ts", "event_date", "direction", "strike", "best_bid", "best_ask"]].rename(columns={"ts": "snap_ts"}).sort_values("snap_ts")
    g["direction"] = g.direction.fillna("")
    m = pd.merge_asof(f.sort_values("ts"), g, left_on="ts", right_on="snap_ts", by=["event_date", "direction", "strike"], tolerance=150, direction="backward")
    matched = m.dropna(subset=["snap_ts"])
    ref = np.where(matched.buy_yes, matched.best_ask, matched.best_bid)
    diff = pd.Series(np.where(matched.buy_yes, matched.p_yes - ref, ref - matched.p_yes), index=matched.index).dropna()
    usd = (matched["size"] * matched.p_yes).reindex(diff.index)
    w1 = (diff.abs() <= 0.0101).mean(); w3 = (diff.abs() <= 0.0301).mean()
    print(f"{fam:5s} prints {len(f):,}; with a snapshot <= 150s before: {len(matched)/len(f):.1%}; of those with a quote on that side {len(diff):,}: "
          f"within 1c of the logged quote {w1:.1%}, within 3c {w3:.1%}; median |diff| {diff.abs().median():.4f}; $-weighted mean diff {np.average(diff, weights=usd):+.4f} "
          f"(positive = print worse for the taker than the logged quote)")
    note(f"F {fam} prints", len(matched) / len(f) > 0.9 and w3 > 0.85, f"{len(matched)/len(f):.0%} of prints have a book, {w1:.0%} within 1c, {w3:.0%} within 3c")

print("\n=== verdict ===")
for n, s, msg in verdict: print(f"{s} {n:18s} {msg}")
