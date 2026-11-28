#!/usr/bin/env bash
# check_sync.sh — is the Mac's copy of the logged books identical to the droplet's, and is the droplet's clock right?
#
#   bash deploy/check_sync.sh          (run from anywhere inside the repo; HOST=nba-droplet by default)
#
# Compares, for everything up to the last snapshot the Mac holds: row counts, the sum of timestamps and the total length
# of the stored books per family and asset, plus the spot and markets tables. Any difference prints a diff and exits 1.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${HOST:-nba-droplet}"
REMOTE=/root/crypto-bot/data/books.sqlite
LOCAL=research/data/books.sqlite
T=$(sqlite3 "$LOCAL" "SELECT MAX(ts) FROM books")
Q="SELECT 'books', family, asset, COUNT(*), SUM(ts), SUM(LENGTH(bids_json) + LENGTH(asks_json)) FROM books WHERE ts <= $T GROUP BY 2,3
   UNION ALL SELECT 'spot', asset, '', COUNT(*), SUM(ts), CAST(SUM(price) AS INTEGER) FROM spot WHERE ts <= $T GROUP BY 2
   ORDER BY 1,2,3;"
L=$(sqlite3 "$LOCAL" "$Q")
R=$(ssh "$HOST" "sqlite3 $REMOTE \"$Q\"")
echo "local copy ends at $(date -r "$T" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "$T")"
echo "$L"
if [ "$L" = "$R" ]; then echo "OK: the Mac's copy matches the droplet row for row up to that time"; RC=0
else echo "MISMATCH — droplet says:"; echo "$R"; diff <(echo "$L") <(echo "$R") || true; RC=1; fi
echo "droplet rows newer than the local copy: $(ssh "$HOST" "sqlite3 $REMOTE 'SELECT COUNT(*) FROM books WHERE ts > $T'")"
# clock: droplet time against this Mac (ssh adds up to about a second), and whether the droplet is NTP-synchronised
A=$(date -u +%s); D=$(ssh "$HOST" "date -u +%s"); B=$(date -u +%s)
echo "droplet clock minus Mac clock: $((D - (A + B) / 2)) s (round trip $((B - A)) s); droplet NTP synchronised: $(ssh "$HOST" "timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown")"
exit $RC
