#!/bin/bash
DEV=A6F3E849-1947-5202-9AD1-9C881CA58EEF; APP=com.iosllmbenchmark.benchmarkapp
EX=~/code/coreai/coreai-models/exports; APP_PATH=/Users/USER/bench-dd/Build/Products/Release-iphoneos/BenchmarkApp.app
RES=/Users/USER/Downloads/ios-llm-benchmark/results/raw/2026-06-24-coreai-iphone/runs.txt
: > $RES
echo "=== install updated app ===" | tee -a $RES
xcrun devicectl device install app --device $DEV "$APP_PATH" 2>&1 | tail -1 | tee -a $RES; sleep 3
ITEMS="
deepseek-r1-1.5b|deepseek_r1_1_5b
tinyswallow-1.5b|tinyswallow_1_5b
vibethinker-1.5b|vibethinker_1_5b
"
echo "$ITEMS" | while IFS='|' read -r ID NM; do
  [ -z "$ID" ] && continue
  for be in ane gpu; do
    if [ "$be" = ane ]; then bundle="${NM}_ane_pure4bit"; folder="${NM}_ane"; else bundle="${NM}_gpu"; folder="${NM}_gpu"; fi
    [ -f "$EX/$bundle/metadata.json" ] || { echo "##### core-ai/${ID}-${be}: bundle not ready, skip" | tee -a $RES; continue; }
    echo "##### core-ai/${ID}-${be} #####" | tee -a $RES
    xcrun devicectl device copy to --device $DEV --domain-type appDataContainer --domain-identifier $APP --source "$EX/$bundle" --destination "Documents/CoreAIModels/$folder" >/dev/null 2>&1 && echo "  pushed $folder" | tee -a $RES || echo "  PUSH FAIL $folder" | tee -a $RES
    for run in 1 2 3; do
      line=$(timeout 400 xcrun devicectl device process launch --console --terminate-existing --device $DEV $APP -- \
        --yardstick-autorun --runtime core-ai --model-id "core-ai/${ID}-${be}" --task short-chat --runs 1 </dev/null 2>&1 \
        | grep -E "YARDSTICK_RUN_OK|YARDSTICK_RUN_FAIL|YARDSTICK_FATAL|not_in_catalog|No space|signal")
      echo "  run$run :: $line" | tee -a $RES; sleep 8
    done
  done
done
echo "DONE" | tee -a $RES
