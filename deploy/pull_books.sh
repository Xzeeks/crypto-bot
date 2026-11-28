#!/usr/bin/env bash
# pull_books.sh — bring the droplet's logged order books to the Mac, incrementally.
#
#   deploy/pull_books.sh            (run from anywhere inside the repo; HOST=nba-droplet by default)
#
# First run (no local research/data/books.sqlite): full consistent snapshot (.backup), gzipped.
# Later runs: only rows newer than the local copy (10-minute overlap, INSERT OR REPLACE), plus the whole small markets table
# (its resolve_ts / last_seen fields are updated in place by the logger). The droplet logger keeps running throughout.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${HOST:-nba-droplet}"
REMOTE=/root/crypto-bot/data/books.sqlite
LOCAL=research/data/books.sqlite
mkdir -p research/data

if [ ! -f "$LOCAL" ]; then
  echo "no local copy: pulling a full snapshot"
  ssh "$HOST" "rm -f /tmp/books_snap.sqlite /tmp/books_snap.sqlite.gz && sqlite3 $REMOTE \".timeout 30000\" \".backup /tmp/books_snap.sqlite\" && gzip -9 -f /tmp/books_snap.sqlite && ls -lh /tmp/books_snap.sqlite.gz"
  rsync --partial --progress "$HOST:/tmp/books_snap.sqlite.gz" research/data/
  gunzip -f research/data/books_snap.sqlite.gz
  mv research/data/books_snap.sqlite "$LOCAL"
  ssh "$HOST" "rm -f /tmp/books_snap.sqlite.gz"
else
  T=$(sqlite3 "$LOCAL" "SELECT COALESCE(MAX(ts),0) - 600 FROM books")
  echo "local copy ends at $(date -r $((T + 600)) '+%Y-%m-%d %H:%M' 2>/dev/null || echo $((T + 600))): pulling newer rows"
  ssh "$HOST" "rm -f /tmp/books_inc.sqlite /tmp/books_inc.sqlite.gz && sqlite3 $REMOTE" <<SQL
.timeout 30000
ATTACH '/tmp/books_inc.sqlite' AS inc;
CREATE TABLE inc.books AS SELECT * FROM books WHERE ts > $T;
CREATE TABLE inc.spot AS SELECT * FROM spot WHERE ts > $T;
CREATE TABLE inc.markets AS SELECT * FROM markets;
SQL
  ssh "$HOST" "gzip -9 -f /tmp/books_inc.sqlite && ls -lh /tmp/books_inc.sqlite.gz"
  rsync --partial --progress "$HOST:/tmp/books_inc.sqlite.gz" research/data/
  gunzip -f research/data/books_inc.sqlite.gz
  # logger v3 added markets.gone_ts: make sure both sides have it before the merge (harmless if already there)
  sqlite3 "$LOCAL" "ALTER TABLE markets ADD COLUMN gone_ts INTEGER" 2>/dev/null || true
  sqlite3 research/data/books_inc.sqlite "ALTER TABLE markets ADD COLUMN gone_ts INTEGER" 2>/dev/null || true
  sqlite3 "$LOCAL" <<SQL
ATTACH 'research/data/books_inc.sqlite' AS inc;
BEGIN;
INSERT OR REPLACE INTO books SELECT * FROM inc.books;
INSERT OR REPLACE INTO spot SELECT * FROM inc.spot;
INSERT OR REPLACE INTO markets SELECT * FROM inc.markets;
COMMIT;
SQL
  rm -f research/data/books_inc.sqlite
  ssh "$HOST" "rm -f /tmp/books_inc.sqlite.gz"
fi
sqlite3 "$LOCAL" "SELECT family, asset, COUNT(*), date(MIN(ts),'unixepoch'), datetime(MAX(ts),'unixepoch') FROM books GROUP BY 1,2"
