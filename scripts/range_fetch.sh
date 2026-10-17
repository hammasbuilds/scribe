#!/usr/bin/env bash
# Fetch a large file in byte-range chunks, resuming an existing partial file.
#
# Why: a single long connection to dumps.wikimedia.org was throttled to ~21 KB/s
# here (12 h for 455 MB), while ranged requests to the same URL ran at ~585 KB/s.
#
# Usage: range_fetch.sh URL OUT [CHUNK_BYTES] [EXPECTED_TOTAL]
# The total is read from a HEAD request; pass EXPECTED_TOTAL only to override it.
set -u

URL="$1"; OUT="$2"; CHUNK=${3:-8388608}; TOTAL=${4:-0}

if [ "$TOTAL" -eq 0 ]; then
  # Follow redirects: Zenodo and S3-backed hosts redirect before serving.
  TOTAL=$(curl -sIL --max-time 60 "$URL" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2}END{gsub(/\r/,"",v);print v+0}')
fi
if [ "${TOTAL:-0}" -le 0 ]; then
  echo "could not determine size from HEAD; pass EXPECTED_TOTAL" >&2
  exit 2
fi
echo "total $TOTAL bytes, chunk $CHUNK"

mkdir -p "$(dirname "$OUT")"
fails=0
while :; do
  HAVE=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  if [ "$HAVE" -gt "$TOTAL" ]; then
    # A chunk that came back as a 200 (or an error page) appends more than was
    # asked for. Truncating is correct: the bytes past the end are not the file.
    echo "trimming $((HAVE - TOTAL)) stray bytes"
    truncate -s "$TOTAL" "$OUT"
    HAVE=$TOTAL
  fi
  [ "$HAVE" -ge "$TOTAL" ] && { echo "done: $HAVE/$TOTAL"; break; }

  END=$((HAVE + CHUNK - 1)); [ "$END" -ge "$TOTAL" ] && END=$((TOTAL - 1))
  WANT=$((END - HAVE + 1))
  TMP="${OUT}.part"
  # --fail so an HTTP error is an error, not a body; -w so the status is checkable.
  CODE=$(curl -sL --fail --max-time 300 -r "${HAVE}-${END}" -o "$TMP" -w "%{http_code}" "$URL" || echo 000)
  GOT=$(stat -c%s "$TMP" 2>/dev/null || echo 0)
  if [ "$CODE" = "206" ] && [ "$GOT" -eq "$WANT" ]; then
    cat "$TMP" >> "$OUT"; rm -f "$TMP"; fails=0
    NOW=$(stat -c%s "$OUT")
    echo "$(date +%H:%M:%S) $NOW/$TOTAL ($((NOW * 100 / TOTAL))%)"
  else
    rm -f "$TMP"
    fails=$((fails + 1))
    echo "chunk at $HAVE rejected (http $CODE, $GOT of $WANT bytes); retry $fails"
    [ "$fails" -ge 8 ] && { echo "giving up at $HAVE/$TOTAL" >&2; exit 1; }
    sleep $((fails * 3))
  fi
done
