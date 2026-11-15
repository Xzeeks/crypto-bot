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
