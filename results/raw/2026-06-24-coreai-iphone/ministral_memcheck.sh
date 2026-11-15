#!/usr/bin/env bash
DEV=A6F3E849-1947-5202-9AD1-9C881CA58EEF; APP=com.iosllmbenchmark.benchmarkapp
SRC=~/code/litertlm-convert/out/ministral3-3b-boctav4-extemb
DEST="Documents/models/litert-lm/litert-local__Ministral-3-3B"
RES=~/Downloads/ios-llm-benchmark/results/raw/2026-06-24-coreai-iphone/extemb-isolation
SCR="<work>/min3b"
mkdir -p "$SCR"; cp "$SRC/model.litertlm" "$SCR/model.litertlm"
echo "=== push Ministral-3B extemb ($(ls -la $SCR/model.litertlm | awk '{print $5}') bytes) ==="
xcrun devicectl device copy to --device $DEV --domain-type appDataContainer --domain-identifier $APP \
  --source "$SCR" --destination "$DEST" 2>&1 | tail -1
echo "on-device: $(xcrun devicectl device info files --device $DEV --domain-type appDataContainer --domain-identifier $APP --subdirectory "$DEST" 2>/dev/null | grep -i litertlm | head -1)"
for run in 1 2 3; do
  f="$RES/console_ministral3b_FULL_${run}.txt"
  echo "----- run $run (full console) -----"
  timeout 360 xcrun devicectl device process launch --console --terminate-existing --device $DEV $APP -- \
    --yardstick-autorun --runtime litert-lm --model-id "litert-local/ministral3-3b" --task short-chat --runs 1 </dev/null > "$f" 2>&1
  echo "  exit=$? lines=$(wc -l < "$f")"
  sleep 6
done
echo "MINISTRAL_MEMCHECK_DONE"
