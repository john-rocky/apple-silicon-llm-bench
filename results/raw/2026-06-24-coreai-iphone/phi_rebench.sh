#!/usr/bin/env bash
DEV=A6F3E849-1947-5202-9AD1-9C881CA58EEF; APP=com.iosllmbenchmark.benchmarkapp
PHI=~/code/litertlm-convert/out/_litertcomm/phi-4-mini/Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm
DEST="Documents/models/litert-lm/litert-local__Qwen3-1.7B-int4"   # reuse an existing litert id's dir
RES=~/Downloads/ios-llm-benchmark/results/raw/2026-06-24-coreai-iphone/extemb-isolation
SCR="<work>/phi"
SUM="$RES/PHI_REBENCH.txt"; : > "$SUM"
echo "Phi model size: $(ls -lh $PHI | awk '{print $5}')" | tee -a "$SUM"
mkdir -p "$SCR"; cp "$PHI" "$SCR/model.litertlm"
echo "=== side-load Phi-4-mini int8 (reusing qwen3-1.7b id dir) ===" | tee -a "$SUM"
xcrun devicectl device copy to --device $DEV --domain-type appDataContainer --domain-identifier $APP \
  --source "$SCR" --destination "$DEST" 2>&1 | tail -1
for run in 1 2 3; do
  f="$RES/console_phi_${run}.txt"
  timeout 420 xcrun devicectl device process launch --console --terminate-existing --device $DEV $APP -- \
    --yardstick-autorun --runtime litert-lm --model-id "litert-local/qwen3-1.7b-int4" --task short-chat --runs 1 </dev/null > "$f" 2>&1
  v=$(grep -hoE "YARDSTICK_RUN_OK[^\"]*decode_tok_s=[0-9.]*|Cannot allocate memory|YARDSTICK_RUN_FAIL run=1 error=[^\"]{0,50}|peak_mb=[0-9]*" "$f" | head -2 | tr '\n' ' ')
  echo "  Phi-4-mini run$run :: ${v:-<no verdict>}" | tee -a "$SUM"
  sleep 6
done
echo "PHI_REBENCH_DONE" | tee -a "$SUM"
