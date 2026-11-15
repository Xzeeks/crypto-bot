#!/usr/bin/env python3
"""book_logger.py — record the Polymarket daily "<asset> above $X" ladder's order books plus Binance spot, on a timer.

Runs as a systemd service on the droplet. Dumb by design: it only records. All rules live in research/scripts.

    python logger/book_logger.py --db /root/crypto-bot/data/books.sqlite --assets ethereum,bitcoin --interval 120

Every tick, for each asset and for each event resolving within the next 30h (today's and tomorrow's noon ET):
  * refresh the event's strike list from gamma if older than 10 min (new events appear ~12:00 ET the day before)
  * fetch the CLOB book for every strike's Yes token; store top 10 levels each side as JSON + best bid/ask/sizes
  * fetch Binance spot once per tick (data-api.binance.vision mirror)
Tables
  markets(asset, event_date, strike, condition_id, yes_token, no_token, resolve_ts, first_seen, last_seen)
  spot(ts, asset, price)
  books(ts, asset, event_date, strike, resolve_ts, spot, best_bid, best_bid_sz, best_ask, best_ask_sz, bids_json, asks_json)
Loop never dies on a bad request: errors are logged and the tick continues.
"""
import argparse, json, logging, sqlite3, time, re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests

GAMMA = "https://gamma-api.polymarket.com"; CLOB = "https://clob.polymarket.com"
BINANCE = "https://data-api.binance.vision/api/v3/ticker/price"
NY = ZoneInfo("America/New_York")
SYMBOL = {"ethereum": "ETHUSDT", "bitcoin": "BTCUSDT"}
STRIKE_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")
log = logging.getLogger("book_logger")
S = requests.Session(); S.headers["User-Agent"] = "crypto-bot-book-logger/0.1"


def get(url, params=None, timeout=20):
    r = S.get(url, params=params, timeout=timeout); r.raise_for_status(); return r.json()


def slug(asset, d):
    return f"{asset}-above-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"


def resolve_ts(d):
    return int(datetime(d.year, d.month, d.day, 12, 0, tzinfo=NY).timestamp())


def init(db):
    con = sqlite3.connect(db, timeout=30)
    con.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS markets(asset TEXT, event_date TEXT, strike REAL, condition_id TEXT, yes_token TEXT, no_token TEXT,
        resolve_ts INTEGER, first_seen INTEGER, last_seen INTEGER, PRIMARY KEY(asset, event_date, strike));
    CREATE TABLE IF NOT EXISTS spot(ts INTEGER, asset TEXT, price REAL, PRIMARY KEY(ts, asset));
    CREATE TABLE IF NOT EXISTS books(ts INTEGER, asset TEXT, event_date TEXT, strike REAL, resolve_ts INTEGER, spot REAL,
        best_bid REAL, best_bid_sz REAL, best_ask REAL, best_ask_sz REAL, bids_json TEXT, asks_json TEXT,
        PRIMARY KEY(ts, asset, event_date, strike));
    CREATE INDEX IF NOT EXISTS books_asset_ts ON books(asset, ts);
    """)
    return con


def refresh_markets(con, asset, now):
    today = datetime.fromtimestamp(now, NY).date()
    for d in (today, today + timedelta(days=1)):
        rts = resolve_ts(d)
        if rts < now: continue
        try:
            ev = get(f"{GAMMA}/events", {"slug": slug(asset, d)})
        except Exception as e:
            log.warning("gamma %s %s: %s", asset, d, e); continue
        if not ev: continue
        for m in ev[0].get("markets", []):
            q = m.get("question") or m.get("groupItemTitle") or ""
            mm = STRIKE_RE.search(q)
            toks = m.get("clobTokenIds"); toks = json.loads(toks) if isinstance(toks, str) else (toks or [])
            if not mm or len(toks) < 2: continue
            outs = m.get("outcomes"); outs = json.loads(outs) if isinstance(outs, str) else (outs or ["Yes", "No"])
            yi = outs.index("Yes") if "Yes" in outs else 0
            con.execute("""INSERT INTO markets VALUES (?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(asset, event_date, strike) DO UPDATE SET last_seen=excluded.last_seen""",
                        (asset, d.isoformat(), float(mm.group(1).replace(",", "")), m.get("conditionId"), toks[yi], toks[1 - yi], rts, now, now))
    con.commit()


def fetch_spot(asset):
    try:
        return float(get(BINANCE, {"symbol": SYMBOL[asset]}, timeout=10)["price"])
    except Exception as e:
        log.warning("spot %s: %s", asset, e); return None


def fetch_book(token):
    b = get(f"{CLOB}/book", {"token_id": token}, timeout=15)
    bids = sorted(((float(x["price"]), float(x["size"])) for x in b.get("bids", [])), reverse=True)[:10]
    asks = sorted(((float(x["price"]), float(x["size"])) for x in b.get("asks", [])))[:10]
    return bids, asks


def tick(con, assets, now):
    for asset in assets:
        spot = fetch_spot(asset)
        if spot is not None:
            con.execute("INSERT OR REPLACE INTO spot VALUES (?,?,?)", (now, asset, spot))
        rows = con.execute("""SELECT event_date, strike, yes_token, resolve_ts FROM markets
                              WHERE asset=? AND resolve_ts > ? AND resolve_ts < ?""", (asset, now - 3600, now + 30 * 3600)).fetchall()
        n = 0
        for event_date, strike, yes_token, rts in rows:
            try:
                bids, asks = fetch_book(yes_token)
            except Exception as e:
                log.warning("book %s %s %s: %s", asset, event_date, strike, e); continue
            bb, bbs = (bids[0] if bids else (None, None)); ba, bas = (asks[0] if asks else (None, None))
            con.execute("INSERT OR REPLACE INTO books VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (now, asset, event_date, strike, rts, spot, bb, bbs, ba, bas, json.dumps(bids), json.dumps(asks)))
            n += 1
            time.sleep(0.15)
        con.commit()
        log.info("%s spot=%s books=%d", asset, spot, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--assets", default="ethereum,bitcoin")
    ap.add_argument("--interval", type=int, default=120, help="seconds between ticks")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    assets = [x.strip() for x in a.assets.split(",") if x.strip()]
    con = init(a.db)
    last_refresh = 0
    while True:
        t0 = time.time(); now = int(t0)
        try:
            if now - last_refresh > 600:
                for asset in assets: refresh_markets(con, asset, now)
                last_refresh = now
            tick(con, assets, now)
        except Exception as e:
            log.exception("tick failed: %s", e)
        if a.once: break
        time.sleep(max(1.0, a.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
