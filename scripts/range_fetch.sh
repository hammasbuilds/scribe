#!/usr/bin/env bash
# Fetch a large file in byte-range chunks, appending to an existing partial file.
# A single long connection gets throttled to ~20 KB/s here; ranged chunks run ~30x faster.
set -u
URL="$1"; OUT="$2"; TOTAL="$3"; CHUNK=${4:-8388608}
while :; do
  HAVE=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  [ "$HAVE" -ge "$TOTAL" ] && { echo "done: $HAVE/$TOTAL"; break; }
  END=$((HAVE + CHUNK - 1)); [ "$END" -ge "$TOTAL" ] && END=$((TOTAL - 1))
  if curl -s --max-time 180 -r "${HAVE}-${END}" "$URL" >> "$OUT"; then
    NOW=$(stat -c%s "$OUT")
    echo "$(date +%H:%M:%S) $NOW/$TOTAL ($((NOW * 100 / TOTAL))%)"
    [ "$NOW" -le "$HAVE" ] && { echo "no progress; retrying"; sleep 5; }
  else
    echo "chunk failed at $HAVE; retrying"; sleep 5
  fi
done
