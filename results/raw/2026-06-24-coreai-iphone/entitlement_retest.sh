#!/usr/bin/env bash
DEV=A6F3E849-1947-5202-9AD1-9C881CA58EEF; APP=com.iosllmbenchmark.benchmarkapp
RES=~/Downloads/ios-llm-benchmark/results/raw/2026-06-24-coreai-iphone/extemb-isolation
OUT=~/code/litertlm-convert/out
SCR="<work>/llama3b"
SUM="$RES/ENTITLEMENT_RETEST.txt"; : > "$SUM"
# side-load Llama-3B extemb (Ministral already on device)
mkdir -p "$SCR"; cp "$OUT/llama32-3b-official-extemb/model.litertlm" "$SCR/model.litertlm"
echo "=== side-load Llama-3B extemb (2.21G) ===" | tee -a "$SUM"
xcrun devicectl device copy to --device $DEV --domain-type appDataContainer --domain-identifier $APP \
  --source "$SCR" --destination "Documents/models/litert-lm/litert-local__Llama-3.2-3B" 2>&1 | tail -1

run_model() { # <label> <model-id>
  local label="$1" mid="$2"
  echo "===== $label =====" | tee -a "$SUM"
  for run in 1 2 3; do
    local f="$RES/console_RETEST_${label}_${run}.txt"
    timeout 360 xcrun devicectl device process launch --console --terminate-existing --device $DEV $APP -- \
      --yardstick-autorun --runtime litert-lm --model-id "$mid" --task short-chat --runs 1 </dev/null > "$f" 2>&1
    local v
    v=$(grep -hoE "YARDSTICK_RUN_OK[^\"]*decode_tok_s=[0-9.]*|Cannot allocate memory|YARDSTICK_RUN_FAIL run=1 error=[^\"]{0,60}|Failed to create engine" "$f" | head -1)
    echo "  $label run$run :: ${v:-<no verdict>}" | tee -a "$SUM"
    sleep 6
  done
}
run_model "Ministral-3B" "litert-local/ministral3-3b"
run_model "Llama-3B"     "litert-local/llama-3.2-3b"
echo "RETEST_DONE" | tee -a "$SUM"
