#!/usr/bin/env python3
"""book_logger.py — record Polymarket daily crypto ladders' order books plus Binance spot, on a timer.

Runs as a systemd service on the droplet. Dumb by design: it only records. All rules live in research/scripts.

    python logger/book_logger.py --db /root/crypto-bot/data/books.sqlite --assets ethereum,bitcoin --interval 120

Two event families per asset, today's and tomorrow's (New York dates):
  above  "<Asset> above ___ on <date>?"          strikes "$X";   resolves on the 1m close at 12:00 ET
  hit    "What price will <Asset> hit on <date>?" outcomes "↑ X" / "↓ X"; one-touch on 1m high/low, window ends 23:59 ET
Every tick: refresh strike lists from gamma if older than 10 min; fetch the CLOB book for every outcome's Yes token
(top --levels levels each side, 25 by default); fetch Binance spot once per asset (data-api.binance.vision mirror).
Tables (v2 — tables are recreated once if the v1 schema is found; v1 rows are carried over as family='above')
  markets(family, asset, event_date, direction, strike, condition_id, yes_token, no_token, resolve_ts, created_ts, first_seen, last_seen, gone_ts)

v3 (2026-11-28):
  * 25 book levels per side instead of 10 (the 10-level cut, not price, was bounding half of the paper carry fills).
  * Each asset's rows are stamped with the time that asset's fetch started (before: both assets got the tick's start time,
    so the second asset's books were stamped 7-14 s early). Rows of one asset still share one ts per tick.
  * "Book not found" (404) handling. gone_ts = when the book disappeared; such markets are no longer polled.
      - market already ended           -> gone at once.
      - hit barrier still live         -> gone only on the second tick in a row (a touch), stamped with the first; never when
                                          every book of the asset is 'not found' in the same tick (an exchange problem).
      - close-ladder strike before its print -> never (a stray 404 used to stop it for good).
    hit keeps the old convention too (resolve_ts = gone_ts - 7200, last_seen = gone_ts): evaluators read resolve_ts + 7200
    as the time the book was pulled. For 'above', resolve_ts now always stays the 12:00 ET print; rows the old rule had
    overwritten are repaired once at start-up.
  spot(ts, asset, price)
  books(ts, family, asset, event_date, direction, strike, resolve_ts, spot, best_bid, best_bid_sz, best_ask, best_ask_sz, bids_json, asks_json)
"""
import argparse, json, logging, sqlite3, time, re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests

GAMMA = "https://gamma-api.polymarket.com"; CLOB = "https://clob.polymarket.com"
BINANCE = "https://data-api.binance.vision/api/v3/ticker/price"
NY = ZoneInfo("America/New_York")
SYMBOL = {"ethereum": "ETHUSDT", "bitcoin": "BTCUSDT"}
NUM_RE = re.compile(r"([\d,]+(?:\.\d+)?)")
FAMILIES = {
    "above": dict(slug="{asset}-above-on-{month}-{day}-{year}", resolve=lambda d: int(datetime(d.year, d.month, d.day, 12, 0, tzinfo=NY).timestamp())),
    "hit":   dict(slug="what-price-will-{asset}-hit-on-{month}-{day}-{year}", resolve=lambda d: int((datetime(d.year, d.month, d.day, tzinfo=NY) + timedelta(days=1)).timestamp())),
}
log = logging.getLogger("book_logger")
S = requests.Session(); S.headers["User-Agent"] = "crypto-bot-book-logger/0.3"


def get(url, params=None, timeout=20):
    r = S.get(url, params=params, timeout=timeout); r.raise_for_status(); return r.json()


def slug(family, asset, d):
    return FAMILIES[family]["slug"].format(asset=asset, month=d.strftime("%B").lower(), day=d.day, year=d.year)


def parse_outcome(family, m):
    """-> (direction, strike) or None.  above: '' / $X.  hit: '↑'|'↓' / X."""
    title = (m.get("groupItemTitle") or m.get("question") or "").strip()
    if family == "above":
        mm = re.search(r"\$\s?" + NUM_RE.pattern, m.get("question") or title)
        return ("", float(mm.group(1).replace(",", ""))) if mm else None
    # groupItemTitle is "↑ 2,300" / "↓ 1,700"; question is "Will Ethereum reach $2,300 ..." / "Will Ethereum dip to $1,700 ..."
    for text in (title, m.get("question") or ""):
        t = text.lower()
        d = "↑" if ("↑" in text or "reach" in t or "rise to" in t) else ("↓" if ("↓" in text or "dip" in t or "fall to" in t or "drop to" in t) else None)
        mm = NUM_RE.search(text.replace("$", ""))
        if d and mm:
            return (d, float(mm.group(1).replace(",", "")))
    return None


def iso_ts(s):
    try: return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception: return None


def init(db):
    con = sqlite3.connect(db, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    cols = [r[1] for r in con.execute("PRAGMA table_info(books)").fetchall()]
    if cols and "family" not in cols:   # migrate v1 -> v2
        log.info("migrating v1 schema")
        con.executescript("ALTER TABLE books RENAME TO books_v1; ALTER TABLE markets RENAME TO markets_v1;")
    con.executescript("""
    CREATE TABLE IF NOT EXISTS markets(family TEXT, asset TEXT, event_date TEXT, direction TEXT, strike REAL, condition_id TEXT,
        yes_token TEXT, no_token TEXT, resolve_ts INTEGER, created_ts INTEGER, first_seen INTEGER, last_seen INTEGER,
        PRIMARY KEY(family, asset, event_date, direction, strike));
    CREATE TABLE IF NOT EXISTS spot(ts INTEGER, asset TEXT, price REAL, PRIMARY KEY(ts, asset));
    CREATE TABLE IF NOT EXISTS books(ts INTEGER, family TEXT, asset TEXT, event_date TEXT, direction TEXT, strike REAL, resolve_ts INTEGER,
        spot REAL, best_bid REAL, best_bid_sz REAL, best_ask REAL, best_ask_sz REAL, bids_json TEXT, asks_json TEXT,
        PRIMARY KEY(ts, family, asset, event_date, direction, strike));
    CREATE INDEX IF NOT EXISTS books_fam_asset_ts ON books(family, asset, ts);
    """)
    if cols and "family" not in cols:
        con.executescript("""
        INSERT OR IGNORE INTO markets SELECT 'above', asset, event_date, '', strike, condition_id, yes_token, no_token, resolve_ts, NULL, first_seen, last_seen FROM markets_v1;
        INSERT OR IGNORE INTO books SELECT ts, 'above', asset, event_date, '', strike, resolve_ts, spot, best_bid, best_bid_sz, best_ask, best_ask_sz, bids_json, asks_json FROM books_v1;
        DROP TABLE books_v1; DROP TABLE markets_v1;""")
    if "gone_ts" not in [r[1] for r in con.execute("PRAGMA table_info(markets)").fetchall()]:      # v2 -> v3
        log.info("migrating to v3: adding markets.gone_ts")
        con.execute("ALTER TABLE markets ADD COLUMN gone_ts INTEGER")
        n_ab = n_hit = 0
        for fam, asset, ed, d, k, rts in con.execute("SELECT family, asset, event_date, direction, strike, resolve_ts FROM markets").fetchall():
            nominal = FAMILIES[fam]["resolve"](date.fromisoformat(ed))
            if rts == nominal: continue
            if fam == "above":      # the old rule overwrote the print time with (404 time - 7200): put the print time back
                con.execute("UPDATE markets SET gone_ts=?, resolve_ts=? WHERE family=? AND asset=? AND event_date=? AND direction=? AND strike=?", (rts + 7200, nominal, fam, asset, ed, d, k)); n_ab += 1
            else:
                con.execute("UPDATE markets SET gone_ts=? WHERE family=? AND asset=? AND event_date=? AND direction=? AND strike=?", (rts + 7200, fam, asset, ed, d, k)); n_hit += 1
        log.info("v3 migration: %d close-ladder print times restored, %d touch barriers marked gone", n_ab, n_hit)
    con.commit()
    return con


def refresh_markets(con, asset, now):
    today = datetime.fromtimestamp(now, NY).date()
    for family, spec in FAMILIES.items():
        for d in (today, today + timedelta(days=1)):
            rts = spec["resolve"](d)
            if rts < now: continue
            try:
                ev = get(f"{GAMMA}/events", {"slug": slug(family, asset, d)})
            except Exception as e:
                log.warning("gamma %s %s %s: %s", family, asset, d, e); continue
            if not ev: continue
            n = 0
            for m in ev[0].get("markets", []):
                po = parse_outcome(family, m)
                toks = m.get("clobTokenIds"); toks = json.loads(toks) if isinstance(toks, str) else (toks or [])
                if not po or len(toks) < 2: continue
                outs = m.get("outcomes"); outs = json.loads(outs) if isinstance(outs, str) else (outs or ["Yes", "No"])
                yi = outs.index("Yes") if "Yes" in outs else 0
                con.execute("""INSERT INTO markets(family, asset, event_date, direction, strike, condition_id, yes_token, no_token, resolve_ts, created_ts, first_seen, last_seen)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                               ON CONFLICT(family, asset, event_date, direction, strike) DO UPDATE SET last_seen=excluded.last_seen""",
                            (family, asset, d.isoformat(), po[0], po[1], m.get("conditionId"), toks[yi], toks[1 - yi], rts,
                             iso_ts(m.get("createdAt") or m.get("startDate") or ""), now, now))
                n += 1
            if n == 0: log.warning("%s %s %s: event found but 0 outcomes parsed", family, asset, d)
    con.commit()


def fetch_spot(asset):
    try:
        return float(get(BINANCE, {"symbol": SYMBOL[asset]}, timeout=10)["price"])
    except Exception as e:
        log.warning("spot %s: %s", asset, e); return None


def fetch_book(token, levels=25):
    b = get(f"{CLOB}/book", {"token_id": token}, timeout=15)
    bids = sorted(((float(x["price"]), float(x["size"])) for x in b.get("bids", [])), reverse=True)[:levels]
    asks = sorted(((float(x["price"]), float(x["size"])) for x in b.get("asks", [])))[:levels]
    return bids, asks


def tick(con, assets, levels, miss):
    """miss: {market key: time of its first 'not found' in the current run of 404s} — kept in memory between ticks."""
    for asset in assets:
        now = int(time.time())          # this asset's rows are stamped with the time its fetch starts
        spot = fetch_spot(asset)
        if spot is not None:
            con.execute("INSERT OR REPLACE INTO spot VALUES (?,?,?)", (now, asset, spot))
        rows = con.execute("""SELECT family, event_date, direction, strike, yes_token, resolve_ts FROM markets
                              WHERE asset=? AND gone_ts IS NULL AND resolve_ts > ? AND resolve_ts < ?""", (asset, now - 3600, now + 40 * 3600)).fetchall()
        n = 0; notfound = []
        for family, event_date, direction, strike, yes_token, rts in rows:
            key = (family, asset, event_date, direction, strike)
            try:
                bids, asks = fetch_book(yes_token, levels)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404: notfound.append((key, rts))
                else: log.warning("book %s %s %s %s%s: %s", family, asset, event_date, direction, strike, e)
                continue
            except Exception as e:
                log.warning("book %s %s %s %s%s: %s", family, asset, event_date, direction, strike, e); continue
            miss.pop(key, None)
            bb, bbs = (bids[0] if bids else (None, None)); ba, bas = (asks[0] if asks else (None, None))
            con.execute("INSERT OR REPLACE INTO books VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (now, family, asset, event_date, direction, strike, rts, spot, bb, bbs, ba, bas, json.dumps(bids), json.dumps(asks)))
            n += 1
            time.sleep(0.15)
        # every live book of the asset 'not found' in one tick is an exchange problem, not a set of touches
        outage = n == 0 and sum(1 for _, rts in notfound if now < rts) >= 3
        if outage: log.warning("%s: all %d books returned 404 — treating as an exchange problem, nothing marked", asset, len(notfound))
        for key, rts in notfound:
            family, _, event_date, direction, strike = key
            first = miss.setdefault(key, now)
            if now >= rts: gone = True                                                # the market has ended: its book was pulled
            elif family == "hit" and not outage and first < now: gone = True          # second tick in a row: the barrier was touched at `first`
            else: gone = False
            if not gone:
                (log.info if family == "hit" and not outage else log.warning)("book not found: %s %s %s %s%s (will confirm next tick)", family, asset, event_date, direction, strike)
                continue
            if family == "hit":      # old convention kept for the evaluators: resolve_ts + 7200 = when the book was pulled
                con.execute("UPDATE markets SET gone_ts=?, resolve_ts=?, last_seen=? WHERE family=? AND asset=? AND event_date=? AND direction=? AND strike=?", (first, first - 7200, first) + key)
            else:                    # close ladder: resolve_ts stays the print time
                con.execute("UPDATE markets SET gone_ts=?, last_seen=? WHERE family=? AND asset=? AND event_date=? AND direction=? AND strike=?", (first, first) + key)
            miss.pop(key, None)
            log.info("book gone: %s %s %s %s%s (%s)", family, asset, event_date, direction, strike, "after its end" if now >= rts else "touched")
        con.commit()
        log.info("%s spot=%s books=%d", asset, spot, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--assets", default="ethereum,bitcoin")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--levels", type=int, default=25, help="book levels stored per side")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    assets = [x.strip() for x in a.assets.split(",") if x.strip()]
    con = init(a.db)
    log.info("book_logger v3: levels=%d interval=%ds assets=%s", a.levels, a.interval, ",".join(assets))
    last_refresh = 0; miss = {}
    while True:
        t0 = time.time(); now = int(t0)
        try:
            if now - last_refresh > 600:
                for asset in assets: refresh_markets(con, asset, now)
                last_refresh = now
            tick(con, assets, a.levels, miss)
        except Exception as e:
            log.exception("tick failed: %s", e)
        if a.once: break
        time.sleep(max(1.0, a.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
