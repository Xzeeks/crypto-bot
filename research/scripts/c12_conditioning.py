#!/usr/bin/env python3
"""c12_conditioning.py — do conditions known at entry change a cell's risk, and does the market already price them?

    python research/scripts/c12_conditioning.py --asset ethereum [--data-dir research/data]

Uses census_<a>.duckdb (3 years of 1m klines) for RISK, and <a>_above.duckdb / <a>_hit.duckdb (Polymarket taker fills) for PRICE.

For each cell the "event" is the thing that decides the trade:
  carry    strike 5-8% (ETH) / 4-8% (BTC) below spot at noon ET the day before; event = closes BELOW the strike at noon  (a loss)
  sellDn   ↓ barrier 4-7% below spot with 6h left in the day, not yet touched;  event = touched before midnight ET      (a loss)
  sellUp   ↑ barrier 3-7% above spot with 9h left, not yet touched;             event = touched                         (a loss)
  buyT     barrier 3-5% away (either side) with 18h left, not yet touched;      event = touched                         (a WIN for the buyer)
Conditions (all known at entry): day of week of the event day, previous-24h move, trailing 30-day realized-vol tercile,
price vs its 30-day average, and (touch cells) the day's range so far.

Per condition bucket:
  days3y / rate3y%   3-year frequency of the event (pooled over the cell's distances)
  fdays / implied%   Polymarket fills in the cell on such days: number of days, and the event rate the price paid implies
                     ($-weighted; 1 - price for the three selling cells, price for buyT)
  gap                implied% - rate3y%.  Selling cells: POSITIVE gap = the market charges more than the 3-year risk (good to sell).
                     buyT: NEGATIVE gap = the ticket costs less than the 3-year hit rate (good to buy).
  real%              what actually happened to those fills ($-weighted) — small samples, read with care.
"""
import argparse, math, os
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import duckdb, numpy as np, pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
NY = ZoneInfo("America/New_York")
ap = argparse.ArgumentParser(); ap.add_argument("--asset", default="ethereum"); ap.add_argument("--data-dir", default="research/data")
a = ap.parse_args()
pre = {"ethereum": "eth", "bitcoin": "btc"}[a.asset]
P = lambda n: os.path.join(a.data_dir, n)
CELLS = {
    "carry":  dict(dists=[5, 6, 7, 8] if a.asset == "ethereum" else [4, 5, 6, 7, 8], pct=(5, 8) if a.asset == "ethereum" else (4, 8), hours=(12, 24), sell=True),
    "sellDn": dict(dists=[4, 5, 6, 7], pct=(4, 7), hours=(2, 6), left=6, dirs="↓", sell=True),
    "sellUp": dict(dists=[3, 4, 5, 6, 7], pct=(3, 7), hours=(3, 9), left=9, dirs="↑", sell=True),
    "buyT":   dict(dists=[3, 4, 5], pct=(3, 5), hours=(12, 18), left=18, dirs=None, sell=False),
}

# ---------- klines: census (3y) + whatever the fills DBs add ----------
parts = []
for n in (f"census_{pre}.duckdb", f"{pre}_hit.duckdb", f"{pre}_above.duckdb"):
    if os.path.exists(P(n)):
        d = duckdb.connect(P(n), read_only=True); parts.append(d.execute("SELECT open_ts, high, low, close FROM klines").df()); d.close()
if not parts: raise SystemExit("no kline DBs found in " + a.data_dir)
k = pd.concat(parts).drop_duplicates("open_ts").sort_values("open_ts").set_index("open_ts")
mid = lambda d: int(datetime(d.year, d.month, d.day, tzinfo=NY).timestamp())
noon = lambda d: int(datetime(d.year, d.month, d.day, 12, tzinfo=NY).timestamp())
d0 = datetime.fromtimestamp(k.index[0], NY).date() + timedelta(days=32); d1 = datetime.fromtimestamp(k.index[-1], NY).date() - timedelta(days=1)
alld = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
# daily closes at 00:00 ET for rv30 / 30d average (as known at the START of day D)
first_day = datetime.fromtimestamp(k.index[0], NY).date() + timedelta(days=1)
dl = [first_day + timedelta(days=i) for i in range((d1 - first_day).days + 2)]
dc = pd.Series({d: k.close.get(mid(d), np.nan) for d in dl}).ffill()
rv30 = (np.log(dc).diff().rolling(30).std() * math.sqrt(365) * 100); ma30 = dc.rolling(30).mean()
rv_cut = rv30.loc[alld[0]:alld[-1]].quantile([1 / 3, 2 / 3]).values
def rv_b(v): return "low" if v < rv_cut[0] else ("mid" if v < rv_cut[1] else "high")
def r24_b(r):
    if r is None or np.isnan(r): return None
    return "fell >4%" if r <= -4 else ("fell 2-4%" if r <= -2 else ("flat ±2%" if r < 2 else ("rose 2-4%" if r < 4 else "rose >4%")))
def rng_b(r): return "<2%" if r < 2 else ("2-4%" if r < 4 else ">4%")
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

rows, feats = [], {}
for D in alld:
    t0, t1 = mid(D), mid(D + timedelta(days=1)); day = k.loc[t0:t1 - 60]
    if len(day) < 1300 or np.isnan(rv30.get(D, np.nan)): continue
    base = dict(dow=DOW[D.weekday()], wk="Sat/Sun" if D.weekday() >= 5 else "Mon-Fri", rv=rv_b(rv30[D]), trend="above 30d avg" if dc[D] >= ma30[D] else "below 30d avg")
    for cell, c in CELLS.items():
        te = noon(D - timedelta(days=1)) if cell == "carry" else t1 - c["left"] * 3600
        if te not in k.index: continue
        spot = k.close[te]; prev = k.close.get(te - 86400, np.nan)
        ft = dict(base, r24=r24_b((spot / prev - 1) * 100 if prev == prev else np.nan))
        if cell == "carry":
            fin = k.close.get(noon(D) - 60, np.nan)
            if fin != fin: continue
            ft["rng"] = None; feats[(cell, D)] = ft
            for dist in c["dists"]: rows.append(dict(ft, cell=cell, D=D, dist=dist, ev=fin <= spot / (1 + dist / 100)))
        else:
            past = day.loc[:te]; rest = day.loc[te + 60:]
            ft["rng"] = rng_b((past.high.max() - past.low.min()) / past.close.iloc[0] * 100); feats[(cell, D)] = ft
            sides = [c["dirs"]] if c["dirs"] else ["↓", "↑"]
            for s in sides:
                for dist in c["dists"]:
                    if s == "↓":
                        b = spot * (1 - dist / 100)
                        if past.low.min() > b: rows.append(dict(ft, cell=cell, D=D, dist=dist, ev=rest.low.min() <= b))
                    else:
                        b = spot * (1 + dist / 100)
                        if past.high.max() < b: rows.append(dict(ft, cell=cell, D=D, dist=dist, ev=rest.high.max() >= b))
R = pd.DataFrame(rows)
print(f"##### {a.asset}: risk from {R.D.nunique()} days ({R.D.min()} .. {R.D.max()}); 30d realized-vol terciles: low < {rv_cut[0]:.0f}% <= mid < {rv_cut[1]:.0f}% <= high")

# ---------- fills in each cell, tagged with the same per-day conditions ----------
F = []
def tag(df, cell):
    ok = [(cell, d) in feats for d in df.D]; df = df[ok].copy()
    for col in ("dow", "wk", "rv", "trend", "r24", "rng"): df[col] = [feats[(cell, d)][col] for d in df.D]
    df["cell"] = cell; return df
if os.path.exists(P(f"{pre}_above.duckdb")):
    d = duckdb.connect(P(f"{pre}_above.duckdb"), read_only=True)
    f = d.execute("""SELECT m.event_date AS D, m.strike, m.resolve_ts, m.outcome, f.ts, f.price, f.size, (f.token = m.yes_token) AS is_yes
                     FROM fills f JOIN markets m USING (condition_id) WHERE m.outcome IS NOT NULL AND f.ts < m.resolve_ts""").df(); d.close()
    f["D"] = pd.to_datetime(f.D).dt.date; f["spot"] = ((f.ts // 60) * 60).map(k.close); f = f.dropna(subset=["spot"])
    f["hours"] = (f.resolve_ts - f.ts) / 3600; f["pct"] = (f.spot - f.strike) / f.strike * 100
    c = CELLS["carry"]; f = f[f.pct.between(*c["pct"]) & f.hours.between(*c["hours"])]
    f["q"] = np.where(f.is_yes, f.price, 1 - f.price); f["ev"] = 1 - f.outcome; f["usd"] = f["size"] * f.q
    F.append(tag(f[["D", "q", "ev", "usd"]], "carry"))
if os.path.exists(P(f"{pre}_hit.duckdb")):
    d = duckdb.connect(P(f"{pre}_hit.duckdb"), read_only=True)
    mk = d.execute("SELECT condition_id, event_date AS D, direction, strike, window_start, window_end, outcome, yes_token FROM markets WHERE outcome IS NOT NULL").df()
    f = d.execute("SELECT condition_id, ts, price, size, token FROM fills").df(); d.close()
    tt = {}
    for r in mk.itertuples():      # first touch time per barrier, so prints after the barrier died are excluded
        seg = k.loc[r.window_start:r.window_end - 60]
        hit = seg.index[(seg.low.values <= r.strike) if r.direction == "↓" else (seg.high.values >= r.strike)]
        tt[r.condition_id] = int(hit[0]) if len(hit) else r.window_end
    f = f.merge(mk, on="condition_id"); f["D"] = pd.to_datetime(f.D).dt.date
    f = f[(f.ts >= f.window_start) & (f.ts < f.condition_id.map(tt))]
    f["spot"] = ((f.ts // 60) * 60).map(k.close); f = f.dropna(subset=["spot"])
    f["hours"] = (f.window_end - f.ts) / 3600
    f["pct"] = np.where(f.direction == "↑", (f.strike - f.spot) / f.spot * 100, (f.spot - f.strike) / f.spot * 100)
    f["p_yes"] = np.where(f.token == f.yes_token, f.price, 1 - f.price)
    for cell in ("sellDn", "sellUp", "buyT"):
        c = CELLS[cell]; g = f[f.pct.between(*c["pct"]) & f.hours.between(*c["hours"])]
        if c["dirs"]: g = g[g.direction == c["dirs"]]
        g = g.assign(q=(1 - g.p_yes) if c["sell"] else g.p_yes, ev=g.outcome)
        g = g.assign(usd=g["size"] * g.q)
        F.append(tag(g[["D", "q", "ev", "usd"]], cell))
F = pd.concat(F) if F else pd.DataFrame(columns=["cell"])

FEATURES = [("wk", "weekend vs weekday (event day)", ["Mon-Fri", "Sat/Sun"]), ("dow", "day of week (event day)", DOW),
            ("r24", "move over the 24h before entry", ["fell >4%", "fell 2-4%", "flat ±2%", "rose 2-4%", "rose >4%"]),
            ("rv", "30-day realized vol regime", ["low", "mid", "high"]), ("trend", "price vs 30-day average", ["above 30d avg", "below 30d avg"]),
            ("rng", "day's range so far at entry", ["<2%", "2-4%", ">4%"])]
wavg = lambda v, w: float(np.average(v, weights=w)) if len(v) and w.sum() > 0 else np.nan
for cell, c in CELLS.items():
    r = R[R.cell == cell]; f = F[F.cell == cell] if len(F) else F
    what = "LOSS rate" if c["sell"] else "HIT rate (a win for the buyer)"
    imp_all = (1 - wavg(f.q, f.usd)) * 100 if c["sell"] else wavg(f.q, f.usd) * 100 if len(f) else np.nan
    print(f"\n================ {a.asset} {cell}: event = {what} ================")
    print(f"3-year baseline {r.ev.mean()*100:.2f}% over {r.D.nunique()} days  (by distance: " + ", ".join(f"{d_}% -> {r[r.dist == d_].ev.mean()*100:.1f}%" for d_ in c["dists"]) + ")"
          + (f";  fills: {f.D.nunique()} days, ${f.usd.sum():,.0f}, price-implied {imp_all:.2f}%" if len(f) else ";  no fills"))
    for col, title, order in FEATURES:
        if col == "rng" and cell == "carry": continue
        out = []
        for bkt in order:
            x = r[r[col] == bkt]
            if x.empty: continue
            y = f[f[col] == bkt] if len(f) else f
            imp = ((1 - wavg(y.q, y.usd)) if c["sell"] else wavg(y.q, y.usd)) * 100 if len(y) else np.nan
            real = wavg(y.ev, y.usd) * 100 if len(y) else np.nan
            rate = x.ev.mean() * 100
            out.append({"bucket": bkt, "days3y": x.D.nunique(), "rate3y%": round(rate, 2), "fdays": y.D.nunique() if len(y) else 0, "f_usd": round(y.usd.sum()) if len(y) else 0,
                        "implied%": None if imp != imp else round(imp, 2), "gap": None if imp != imp else round(imp - rate, 2), "real%": None if real != real else round(real, 2)})
        print(f"\n-- {title} --"); print(pd.DataFrame(out).to_string(index=False, na_rep=""))
