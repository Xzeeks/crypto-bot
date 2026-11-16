#!/usr/bin/env python3
"""
c1_eth_above_pull.py — pull Polymarket "Ethereum above $X on <date>" markets, their taker fills,
and Binance ETHUSDT 1-minute klines into one DuckDB.

Run on the Mac (polymarket.com / binance.com are not reachable from the Claude session VM).

    python c1_eth_above_pull.py --days 120 --out research/data/eth_above.duckdb
    python c1_eth_above_pull.py --days 120 --out ... --skip-fills      # markets + klines only
    python c1_eth_above_pull.py --asset bitcoin --days 120 --out research/data/btc_above.duckdb

Tables
  markets  one row per strike market: event_date, strike, condition_id, yes_token, no_token,
           resolve_ts (12:00 ET candle open, epoch s), outcome (1=Yes/0=No/NULL), volume, liquidity
  fills    every taker fill from data-api /trades: ts, price, size, side, token, outcome_label, wallet, tx
  klines   Binance ETHUSDT 1m: open_ts, open, high, low, close, volume

Resolution rule (from market text): Yes iff the Binance ETH/USDT 1-minute candle that OPENS at
12:00 ET on <date> closes strictly above the strike. 12:00 ET is 16:00 UTC in EDT, 17:00 UTC in EST.
"""
import argparse, json, re, sys, time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import duckdb, requests

GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
# data-api.binance.vision is Binance's public market-data mirror (same klines, not geo-blocked; api.binance.com returns 451 from HK)
BINANCE = "https://data-api.binance.vision/api/v3/klines"
NY = ZoneInfo("America/New_York")
S = requests.Session()
S.headers["User-Agent"] = "c1-eth-above/0.1"


def get(url, params=None, tries=5):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 + 2 * i); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa
            if i == tries - 1: raise
            time.sleep(1 + i)


def slug_for(asset: str, d: date) -> str:
    # e.g. ethereum-above-on-july-1-2026 (no zero padding on the day)
    return f"{asset}-above-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"


def resolve_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, 12, 0, tzinfo=NY).timestamp())


STRIKE_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")


def fetch_event(asset, d):
    ev = get(f"{GAMMA}/events", {"slug": slug_for(asset, d)})
    if not ev:
        return None
    return ev[0] if isinstance(ev, list) else ev


def parse_markets(ev, d):
    rows = []
    for m in ev.get("markets", []):
        q = m.get("question") or m.get("groupItemTitle") or ""
        mm = STRIKE_RE.search(q)
        if not mm:
            continue
        strike = float(mm.group(1).replace(",", ""))
        toks = m.get("clobTokenIds")
        toks = json.loads(toks) if isinstance(toks, str) else (toks or [])
        outs = m.get("outcomes")
        outs = json.loads(outs) if isinstance(outs, str) else (outs or ["Yes", "No"])
        prices = m.get("outcomePrices")
        prices = json.loads(prices) if isinstance(prices, str) else (prices or [])
        yes_i = outs.index("Yes") if "Yes" in outs else 0
        no_i = 1 - yes_i
        outcome = None
        if m.get("closed") and prices:
            try:
                py = float(prices[yes_i])
                outcome = 1 if py > 0.99 else (0 if py < 0.01 else None)
            except Exception:
                pass
        rows.append(dict(
            event_date=d.isoformat(), event_id=str(ev.get("id")), market_id=str(m.get("id")),
            question=q, strike=strike, condition_id=m.get("conditionId"),
            yes_token=toks[yes_i] if len(toks) > yes_i else None,
            no_token=toks[no_i] if len(toks) > no_i else None,
            resolve_ts=resolve_ts(d), outcome=outcome,
            volume=float(m.get("volumeNum") or m.get("volume") or 0),
            liquidity=float(m.get("liquidityNum") or m.get("liquidity") or 0),
            created_at=m.get("createdAt"), closed=bool(m.get("closed")),
        ))
    return rows


def fetch_fills(condition_id, page=1000, max_pages=400):
    """Data API v2 (v1 retired 2026-11-02): /v2/trades?condition=...&taker_only=true, cursor-paged, {"data":[...]} envelope."""
    out, cursor = [], None
    for _ in range(max_pages):
        params = {"condition": condition_id, "limit": page, "taker_only": "true", "start": 1}
        if cursor:
            params["cursor"] = cursor
        try:
            r = get(f"{DATA}/v2/trades", params)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                break
            raise
        batch = r.get("data", r) if isinstance(r, dict) else r
        if not batch:
            break
        for t in batch:
            out.append(dict(
                condition_id=condition_id, ts=int(t["timestamp"]), price=float(t["price"]), size=float(t["size"]),
                side=t.get("side"), token=t.get("token_id") or t.get("asset"), outcome_label=t.get("outcome"),
                outcome_index=t.get("outcome_index", t.get("outcomeIndex")),
                wallet=t.get("proxy_wallet") or t.get("proxyWallet"), tx=t.get("transaction_hash") or t.get("transactionHash"),
            ))
        pg = r.get("pagination", {}) if isinstance(r, dict) else {}
        cursor = pg.get("next_cursor")
        if not pg.get("has_more") or not cursor:
            break
    return out


def fetch_klines(symbol, start_ts, end_ts):
    rows, cur = [], start_ts * 1000
    end_ms = end_ts * 1000
    while cur < end_ms:
        b = get(BINANCE, {"symbol": symbol, "interval": "1m", "startTime": cur, "endTime": end_ms, "limit": 1000})
        if not b:
            break
        for k in b:
            rows.append(dict(open_ts=k[0] // 1000, open=float(k[1]), high=float(k[2]), low=float(k[3]),
                             close=float(k[4]), volume=float(k[5])))
        cur = b[-1][0] + 60_000
        if len(b) < 1000:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="ethereum", choices=["ethereum", "bitcoin"])
    ap.add_argument("--days", type=int, default=120, help="how many daily events back from today")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-fills", action="store_true")
    ap.add_argument("--skip-klines", action="store_true")
    a = ap.parse_args()
    symbol = "ETHUSDT" if a.asset == "ethereum" else "BTCUSDT"

    con = duckdb.connect(a.out)
    con.execute("""CREATE TABLE IF NOT EXISTS markets(event_date DATE, event_id TEXT, market_id TEXT PRIMARY KEY,
        question TEXT, strike DOUBLE, condition_id TEXT, yes_token TEXT, no_token TEXT, resolve_ts BIGINT,
        outcome INTEGER, volume DOUBLE, liquidity DOUBLE, created_at TEXT, closed BOOLEAN)""")
    con.execute("""CREATE TABLE IF NOT EXISTS fills(condition_id TEXT, ts BIGINT, price DOUBLE, size DOUBLE, side TEXT,
        token TEXT, outcome_label TEXT, outcome_index INTEGER, wallet TEXT, tx TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS klines(open_ts BIGINT PRIMARY KEY, open DOUBLE, high DOUBLE, low DOUBLE,
        close DOUBLE, volume DOUBLE)""")

    today = datetime.now(timezone.utc).date()
    dates = [today + timedelta(days=1) - timedelta(days=i) for i in range(a.days + 2)]  # include tomorrow's open event
    n_ev = n_mk = 0
    for d in dates:
        ev = fetch_event(a.asset, d)
        if not ev:
            continue
        rows = parse_markets(ev, d)
        if not rows:
            continue
        n_ev += 1; n_mk += len(rows)
        con.executemany("""INSERT OR REPLACE INTO markets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        [(r["event_date"], r["event_id"], r["market_id"], r["question"], r["strike"], r["condition_id"],
                          r["yes_token"], r["no_token"], r["resolve_ts"], r["outcome"], r["volume"], r["liquidity"],
                          r["created_at"], r["closed"]) for r in rows])
        print(f"{d} {len(rows):2d} strikes  vol ${sum(r['volume'] for r in rows):,.0f}", flush=True)
    print(f"events={n_ev} markets={n_mk}")

    if not a.skip_fills:
        done = {r[0] for r in con.execute("SELECT DISTINCT condition_id FROM fills").fetchall()}
        todo = con.execute("SELECT condition_id, question FROM markets WHERE condition_id IS NOT NULL").fetchall()
        for i, (cid, q) in enumerate(todo, 1):
            if cid in done:
                continue
            fl = fetch_fills(cid)
            if fl:
                con.executemany("INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?,?)",
                                [(f["condition_id"], f["ts"], f["price"], f["size"], f["side"], f["token"],
                                  f["outcome_label"], f["outcome_index"], f["wallet"], f["tx"]) for f in fl])
            print(f"[{i}/{len(todo)}] {q[:48]:48s} fills={len(fl)}", flush=True)
            time.sleep(0.2)

    if not a.skip_klines:
        lo, hi = con.execute("SELECT MIN(resolve_ts) - 10*86400, MAX(resolve_ts) + 3600 FROM markets").fetchone()
        have_lo, have_hi = con.execute("SELECT MIN(open_ts), MAX(open_ts) FROM klines").fetchone()
        ranges = [(lo, hi)] if have_lo is None else [(lo, have_lo - 60), (have_hi + 60, hi)]   # backfill older gap + extend forward
        for s, e in ranges:
            if s < e:
                kl = fetch_klines(symbol, s, e)
                con.executemany("INSERT OR REPLACE INTO klines VALUES (?,?,?,?,?,?)",
                                [(k["open_ts"], k["open"], k["high"], k["low"], k["close"], k["volume"]) for k in kl])
                print(f"klines +{len(kl)}")

    # Cross-check resolution rule vs Polymarket outcome
    chk = con.execute("""
        SELECT m.event_date, m.strike, m.outcome, k.close,
               CASE WHEN k.close > m.strike THEN 1 ELSE 0 END AS rule_outcome
        FROM markets m JOIN klines k ON k.open_ts = m.resolve_ts
        WHERE m.outcome IS NOT NULL""").fetchall()
    bad = [r for r in chk if r[2] != r[4]]
    print(f"resolution check: {len(chk)} resolved strikes, {len(bad)} disagree with 12:00 ET 1m-close rule")
    for r in bad[:20]:
        print("  MISMATCH", r)
    con.close()


if __name__ == "__main__":
    main()
