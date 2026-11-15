#!/usr/bin/env bash
DEV=A6F3E849-1947-5202-9AD1-9C881CA58EEF; APP=com.iosllmbenchmark.benchmarkapp
OUT=~/code/litertlm-convert/out
DEST="Documents/models/litert-lm/litert-local__Qwen3-1.7B-int4"
RES=~/Downloads/ios-llm-benchmark/results/raw/2026-06-24-coreai-iphone/extemb-isolation
SCR="<work>"
mkdir -p "$RES" "$SCR/q17_off" "$SCR/q17_on"
cp "$OUT/qwen3-1.7b-boctav4-c4k/model.litertlm"        "$SCR/q17_off/model.litertlm"
cp "$OUT/qwen3-1.7b-boctav4-c4k-EXTEMB/model.litertlm" "$SCR/q17_on/model.litertlm"
SUM="$RES/SUMMARY.txt"; : > "$SUM"

run_variant() { # <OFF|ON> <srcdir>
  local v="$1" src="$2"
  echo "===== variant=$v  (push $(ls -la $src/model.litertlm | awk '{print $5}') bytes) =====" | tee -a "$SUM"
  xcrun devicectl device copy to --device $DEV --domain-type appDataContainer --domain-identifier $APP \
    --source "$src" --destination "$DEST" 2>&1 | tail -1
  echo "  on-device: $(xcrun devicectl device info files --device $DEV --domain-type appDataContainer --domain-identifier $APP --subdirectory "$DEST" 2>/dev/null | grep -i litertlm | head -1)" | tee -a "$SUM"
  for run in 1 2 3 4 5; do
    local f="$RES/console_${v}_${run}.txt"
    timeout 300 xcrun devicectl device process launch --console --terminate-existing --device $DEV $APP -- \
      --yardstick-autorun --runtime litert-lm --model-id "litert-local/qwen3-1.7b-int4" --task short-chat --runs 1 </dev/null > "$f" 2>&1
    local verdict
    verdict=$(grep -hoE "YARDSTICK_RUN_OK[^\"]*decode_tok_s=[0-9.]*|embedding lookup model is not initialized|FAILED_PRECONDITION[^\"]{0,60}|Failed to create engine|YARDSTICK_RUN_FAIL" "$f" | head -1)
    echo "  $v run$run :: ${verdict:-<no verdict — see $(basename $f)>}" | tee -a "$SUM"
    sleep 6
  done
}
run_variant OFF "$SCR/q17_off"
run_variant ON  "$SCR/q17_on"
echo "AB_DONE" | tee -a "$SUM"
