#!/usr/bin/env python3
"""c5_rewards_check.py — do the crypto series pay liquidity rewards, and what does the daily ladder's book look like right now?

    python research/scripts/c5_rewards_check.py

For today's and tomorrow's ETH/BTC "above $X" events: per strike, gamma's reward fields (rewardsMinSize,
rewardsMaxSpread, clobRewards daily rate), plus the live CLOB best bid/ask and depth so we see the book we'd be
quoting into. Also lists the recurring crypto series (5m/15m/hourly up-or-down) with their reward fields.
"""
import json, requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

GAMMA = "https://gamma-api.polymarket.com"; CLOB = "https://clob.polymarket.com"
NY = ZoneInfo("America/New_York")
S = requests.Session(); S.headers["User-Agent"] = "c5-rewards/0.1"

def get(url, **params):
    r = S.get(url, params=params, timeout=30); r.raise_for_status(); return r.json()

def slug(asset, d): return f"{asset}-above-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"

def book(token):
    try:
        b = get(f"{CLOB}/book", token_id=token)
        bids = sorted(((float(x["price"]), float(x["size"])) for x in b.get("bids", [])), reverse=True)
        asks = sorted(((float(x["price"]), float(x["size"])) for x in b.get("asks", [])))
        return bids, asks
    except Exception as e:
        return [], []

def depth(levels, lo, hi):
    return sum(p * s for p, s in levels if lo <= p <= hi)

today = datetime.now(NY).date()
for asset in ("ethereum", "bitcoin"):
    for d in (today, today + timedelta(days=1)):
        ev = get(f"{GAMMA}/events", slug=slug(asset, d))
        if not ev: print(f"{asset} {d}: no event"); continue
        ev = ev[0]
        print(f"\n=== {ev.get('title')}  (vol ${float(ev.get('volume') or 0):,.0f}, liq ${float(ev.get('liquidity') or 0):,.0f}) ===")
        print(f"{'strike':>8} {'rew$/day':>9} {'minSize':>8} {'maxSpr':>7} | {'Yes bid':>8} {'Yes ask':>8} {'bid$0.90-0.99':>14} {'ask$0.90-0.99':>14}")
        for m in sorted(ev["markets"], key=lambda m: m.get("groupItemTitle", "")):
            cr = m.get("clobRewards") or []
            rate = sum(float(x.get("rewardsDailyRate") or 0) for x in cr)
            toks = m.get("clobTokenIds"); toks = json.loads(toks) if isinstance(toks, str) else toks
            bids, asks = book(toks[0]) if toks else ([], [])
            bb = bids[0][0] if bids else None; ba = asks[0][0] if asks else None
            print(f"{m.get('groupItemTitle',''):>8} {rate:>9.2f} {str(m.get('rewardsMinSize')):>8} {str(m.get('rewardsMaxSpread')):>7} | "
                  f"{bb if bb is not None else '-':>8} {ba if ba is not None else '-':>8} {depth(bids,0.90,0.99):>14,.0f} {depth(asks,0.90,0.99):>14,.0f}")

print("\n=== recurring crypto series (reward fields on the currently active market of each) ===")
try:
    series = get(f"{GAMMA}/series", limit=200, active="true")
    for s in series:
        t = (s.get("title") or "").lower()
        if not any(k in t for k in ("bitcoin", "btc", "ethereum", "eth")): continue
        if not any(k in t for k in ("up or down", "updown", "5 min", "15 min", "hour", "above")): continue
        print(f"- {s.get('title')}  slug={s.get('slug')}  recurrence={s.get('recurrence')}")
except Exception as e:
    print("series listing failed:", e)
# spot-check: one active 15m ETH market
for sl in ("eth-updown-15m", "btc-updown-15m", "eth-updown-5m"):
    try:
        ms = get(f"{GAMMA}/markets", slug_contains=sl, active="true", closed="false", limit=3)
        for m in ms[:1]:
            cr = m.get("clobRewards") or []
            print(f"{sl}: {m.get('question')[:60]}  rewards/day={sum(float(x.get('rewardsDailyRate') or 0) for x in cr)}  "
                  f"minSize={m.get('rewardsMinSize')} maxSpread={m.get('rewardsMaxSpread')}  takerFee={m.get('takerBaseFee')} makerFee={m.get('makerBaseFee')}")
    except Exception as e:
        print(sl, "lookup failed:", e)
