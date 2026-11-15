#!/usr/bin/env python3
"""c7_hit_pull.py — backfill Polymarket "What price will <asset> hit on <date>?" one-touch ladders: markets, taker fills, 1m klines.

    python research/scripts/c7_hit_pull.py --asset ethereum --days 200 --out research/data/eth_hit.duckdb
    python research/scripts/c7_hit_pull.py --asset bitcoin  --days 200 --out research/data/btc_hit.duckdb

Outcome "↑ X" = Yes iff some Binance 1m candle HIGH >= X inside the window; "↓ X" = Yes iff some 1m LOW <= X.
Window: from max(00:00 ET on the event date, market createdAt) to 23:59 ET (we use window_end = next midnight ET).
Tables: markets(event_date, market_id, question, direction, strike, condition_id, yes_token, no_token, window_start, window_end,
        outcome, volume, created_at, closed), fills(...), klines(open_ts, open, high, low, close, volume).
Ends with a resolution cross-check of the touch rule against recorded outcomes.
"""
import argparse, json, re, time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import duckdb, requests

GAMMA = "https://gamma-api.polymarket.com"; DATA = "https://data-api.polymarket.com"
BINANCE = "https://data-api.binance.vision/api/v3/klines"
NY = ZoneInfo("America/New_York"); NUM_RE = re.compile(r"([\d,]+(?:\.\d+)?)")
S = requests.Session(); S.headers["User-Agent"] = "c7-hit-pull/0.1"


def get(url, params=None, tries=5):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 429: time.sleep(2 + 2 * i); continue
            r.raise_for_status(); return r.json()
        except Exception:
            if i == tries - 1: raise
            time.sleep(1 + i)


def parse(m):
    title = (m.get("groupItemTitle") or "").strip()
    for text in (title, m.get("question") or ""):
        t = text.lower()
        d = "↑" if ("↑" in text or "reach" in t or "rise to" in t) else ("↓" if ("↓" in text or "dip" in t or "fall to" in t or "drop to" in t) else None)
        mm = NUM_RE.search(text.replace("$", ""))
        if d and mm: return d, float(mm.group(1).replace(",", ""))
    return None


def iso_ts(s):
    try: return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception: return None


def fetch_fills(cid, page=1000, max_pages=200):
    out, off = [], 0
    for _ in range(max_pages):
        b = get(f"{DATA}/trades", {"market": cid, "limit": page, "offset": off, "takerOnly": "true"})
        if not b: break
        out += [(cid, int(t["timestamp"]), float(t["price"]), float(t["size"]), t.get("side"), t.get("asset"), t.get("outcome"), t.get("proxyWallet"), t.get("transactionHash")) for t in b]
        if len(b) < page: break
        off += page
    return out


def fetch_klines(symbol, s, e):
    rows, cur = [], s * 1000
    while cur < e * 1000:
        b = get(BINANCE, {"symbol": symbol, "interval": "1m", "startTime": cur, "endTime": e * 1000, "limit": 1000})
        if not b: break
        rows += [(k[0] // 1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in b]
        cur = b[-1][0] + 60_000
        if len(b) < 1000: break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="ethereum", choices=["ethereum", "bitcoin"])
    ap.add_argument("--days", type=int, default=200); ap.add_argument("--out", required=True)
    ap.add_argument("--skip-fills", action="store_true")
    a = ap.parse_args()
    symbol = "ETHUSDT" if a.asset == "ethereum" else "BTCUSDT"
    con = duckdb.connect(a.out)
    con.execute("""CREATE TABLE IF NOT EXISTS markets(event_date DATE, market_id TEXT PRIMARY KEY, question TEXT, direction TEXT, strike DOUBLE,
        condition_id TEXT, yes_token TEXT, no_token TEXT, window_start BIGINT, window_end BIGINT, outcome INTEGER, volume DOUBLE, created_at TEXT, closed BOOLEAN)""")
    con.execute("CREATE TABLE IF NOT EXISTS fills(condition_id TEXT, ts BIGINT, price DOUBLE, size DOUBLE, side TEXT, token TEXT, outcome_label TEXT, wallet TEXT, tx TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS klines(open_ts BIGINT PRIMARY KEY, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE)")

    today = datetime.now(timezone.utc).date()
    n_ev = 0
    for i in range(a.days + 2):
        d = today + timedelta(days=1) - timedelta(days=i)
        ev = get(f"{GAMMA}/events", {"slug": f"what-price-will-{a.asset}-hit-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"})
        if not ev: continue
        day0 = int(datetime(d.year, d.month, d.day, tzinfo=NY).timestamp()); day1 = day0 + 86400
        rows = []
        for m in ev[0].get("markets", []):
            po = parse(m)
            toks = m.get("clobTokenIds"); toks = json.loads(toks) if isinstance(toks, str) else (toks or [])
            if not po or len(toks) < 2: continue
            outs = m.get("outcomes"); outs = json.loads(outs) if isinstance(outs, str) else (outs or ["Yes", "No"])
            yi = outs.index("Yes") if "Yes" in outs else 0
            prices = m.get("outcomePrices"); prices = json.loads(prices) if isinstance(prices, str) else (prices or [])
            outcome = None
            if m.get("closed") and prices:
                try: py = float(prices[yi]); outcome = 1 if py > 0.99 else (0 if py < 0.01 else None)
                except Exception: pass
            cts = iso_ts(m.get("createdAt") or "")
            rows.append((d.isoformat(), str(m.get("id")), m.get("question"), po[0], po[1], m.get("conditionId"), toks[yi], toks[1 - yi],
                         max(day0, cts or 0), day1, outcome, float(m.get("volumeNum") or m.get("volume") or 0), m.get("createdAt"), bool(m.get("closed"))))
        if rows:
            n_ev += 1
            con.executemany("INSERT OR REPLACE INTO markets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            print(f"{d} {len(rows):2d} outcomes  vol ${sum(r[11] for r in rows):,.0f}", flush=True)
    print(f"events={n_ev}")

    if not a.skip_fills:
        done = {r[0] for r in con.execute("SELECT DISTINCT condition_id FROM fills").fetchall()}
        todo = con.execute("SELECT condition_id, question FROM markets").fetchall()
        for i, (cid, q) in enumerate(todo, 1):
            if cid in done: continue
            fl = fetch_fills(cid)
            if fl: con.executemany("INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?)", fl)
            print(f"[{i}/{len(todo)}] {q[:50]:50s} fills={len(fl)}", flush=True); time.sleep(0.2)

    lo, hi = con.execute("SELECT MIN(window_start) - 86400, MAX(window_end) + 3600 FROM markets").fetchone()
    have_lo, have_hi = con.execute("SELECT MIN(open_ts), MAX(open_ts) FROM klines").fetchone()
    for s, e in ([(lo, hi)] if have_lo is None else [(lo, have_lo - 60), (have_hi + 60, hi)]):
        if s < e:
            kl = fetch_klines(symbol, s, e)
            con.executemany("INSERT OR REPLACE INTO klines VALUES (?,?,?,?,?,?)", kl); print(f"klines +{len(kl)}")

    chk = con.execute("""
        SELECT m.event_date, m.direction, m.strike, m.outcome,
               CASE WHEN m.direction = '↑' THEN MAX(k.high) >= m.strike ELSE MIN(k.low) <= m.strike END AS rule
        FROM markets m JOIN klines k ON k.open_ts >= m.window_start AND k.open_ts < m.window_end
        WHERE m.outcome IS NOT NULL GROUP BY 1,2,3,4""").fetchall()
    bad = [r for r in chk if bool(r[3]) != bool(r[4])]
    print(f"resolution check: {len(chk)} resolved outcomes, {len(bad)} disagree with the 1m high/low touch rule")
    for r in bad[:15]: print("  MISMATCH", r)
    con.close()


if __name__ == "__main__":
    main()
