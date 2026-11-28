# crypto-bot deploy notes

## Droplet (BLR1) — book logger

First time:

    ssh nba-droplet "git clone git@github.com:Xzeeks/crypto-bot.git /root/crypto-bot && cd /root/crypto-bot && python3 -m venv .venv && .venv/bin/pip install -q requests && mkdir -p data && cp deploy/crypto-book-logger.service /etc/systemd/system/ && systemctl daemon-reload && systemctl enable --now crypto-book-logger && sleep 20 && systemctl status crypto-book-logger --no-pager | head -12"

(Set the GitHub remote first on the Mac: `gh repo create Xzeeks/crypto-bot --private --source=. --push`.)

Check it's logging:

    ssh nba-droplet "sqlite3 /root/crypto-bot/data/books.sqlite 'SELECT asset, COUNT(*), MIN(ts), MAX(ts) FROM books GROUP BY 1' && journalctl -u crypto-book-logger -n 5 --no-pager"

Update after a code change (same workflow as nba-bot):

    ssh nba-droplet "cd /root/crypto-bot && git pull && systemctl restart crypto-book-logger"

## Pull the books to the Mac for evaluation

    scp nba-droplet:/root/crypto-bot/data/books.sqlite research/data/books.sqlite
    python research/scripts/c6_shadow_eval.py research/data/books.sqlite --fills research/data/eth_above.duckdb

Re-pull fills first (`c1_eth_above_pull.py --days 30 --out research/data/eth_above.duckdb`) so resolved outcomes and
taker prints cover the logged days; the evaluator uses provisional outcomes from spot otherwise.

Disk: ~22 strikes × 30 ticks/h × 24h ≈ 16k rows/day ≈ 10 MB/day. Add `books.sqlite` to the nba-bot backup rotation
or prune rows older than 60 days once the shadow period is over.

## Logger v3 (2026-11-28)

25 book levels per side (was 10), each asset stamped at its own fetch time, safer handling of "book not found", and a new
`markets.gone_ts` column (added automatically at start-up; `pull_books.sh` adds it to the local copy too). Deploy:

    ssh nba-droplet "sqlite3 /root/crypto-bot/data/books.sqlite '.backup /root/crypto-bot/data/books_pre_v3.sqlite' && cd /root/crypto-bot && git pull -q && systemctl restart crypto-book-logger && sleep 150 && journalctl -u crypto-book-logger -n 12 --no-pager"

Avoid restarting within a few minutes of 12:00 ET (the close-ladder print).

## Routine checks

    deploy/pull_books.sh                       # bring new snapshots to the Mac
    bash deploy/check_sync.sh                  # Mac copy == droplet copy, droplet clock
    python research/scripts/c10_logger_audit.py research/data/books.sqlite --asset ethereum --hit research/data/eth_hit.duckdb --above research/data/eth_above.duckdb

Always run the two pull scripts (`c1_eth_above_pull.py`, `c7_hit_pull.py`) before the audit or a shadow review: they refetch
any market until it has been pulled after its end, re-list missing or unresolved days, and repair the last stored candle.

