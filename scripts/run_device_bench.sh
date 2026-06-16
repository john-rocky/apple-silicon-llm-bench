#!/usr/bin/env bash
# Drive the same-conditions short-chat benchmark on a connected iPhone, headless.
#
# Reproducible protocol used for the LiteRT-LM / Qwen3 + Gemma comparison. Every
# (runtime, model) is run THREE times as SEPARATE cold launches (fresh process each
# → model reloaded → cold), matching "median of 3 cold runs". Same conditions for
# all: short-chat task, maxTokens 128, greedy (temp 0 / top-p 1). LiteRT-LM now caps
# its output at 128 in MediaPipeRuntime, so every runtime generates the same budget.
#
# Prereq: build + install the app once via Xcode (it needs your signing):
#     cd ios/BenchmarkApp && ./scripts/bootstrap.sh   # clones LiteRT-LM v0.13.1 etc.
#     open BenchmarkApp.xcodeproj                      # select the iPhone, ⌘R
#   Build in RELEASE for representative numbers (Product > Scheme > Edit Scheme >
#   Run > Build Configuration: Release), per methodology/fairness-rules.md #7. The
#   inference cores are prebuilt-release xcframeworks, so the Swift-layer Debug/Release
#   gap is small, but Release is the honest default.
#
# Usage:
#     UDID=<device-udid> ./scripts/run_device_bench.sh            # run the matrix
#     UDID=<device-udid> ./scripts/run_device_bench.sh collect    # copy JSONL off-device
#   Find the UDID with:  xcrun devicectl list devices
#
# After collecting, verify each JSON's "coldRun": true and "initialThermalState":
# "nominal" (re-run any that drifted), then split into
# results/raw/<device>-<runtime>-<model>-short-chat-runN.jsonl and regenerate tables.
set -uo pipefail
UDID="${UDID:?set UDID=<device-udid> — see: xcrun devicectl list devices}"
APP=com.iosllmbenchmark.benchmarkapp
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-360}"   # raise if a model still needs to download

# (runtime-kind | model-id). core-ai needs its exported bundle side-loaded into
# Documents/CoreAIModels/<id>/ first (metadata.json + .aimodel + tokenizer/); without
# it the run reports a clear "bundle not found" failure (kept in the table, rule 4).
JOBS=(
  "litert-lm|litert-community/Qwen3-0.6B"
  "mlx-swift|mlx-community/Qwen3-0.6B-4bit"
  "coreml-llm|coreml-llm/qwen3-0.6b"
  "core-ai|core-ai/qwen3-0.6b-gpu"
  "core-ai|core-ai/qwen3-0.6b-ane"
  "litert-lm|litert-community/gemma-4-E2B-it-litert-lm"
  "mlx-swift|mlx-community/gemma-4-e2b-it-4bit"
  "llama.cpp|unsloth/gemma-4-E2B-it-GGUF/Q4_K_M"
  "coreml-llm|coreml-llm/gemma4-e2b"
)

run_matrix() {
  for job in "${JOBS[@]}"; do
    local RT="${job%%|*}" MODEL="${job##*|}"
    echo "########## $RT  $MODEL ##########"
    for run in 1 2 3; do
      echo "----- $RT run$run -----"
      timeout "$PER_RUN_TIMEOUT" xcrun devicectl device process launch --console --terminate-existing \
        --device "$UDID" "$APP" -- \
        --yardstick-autorun --runtime "$RT" --model-id "$MODEL" --task short-chat --runs 1 2>&1 \
        | grep -iE "YARDSTICK_RUN_OK|YARDSTICK_RUN_FAIL|signal 1[15]|not_in_catalog|YARDSTICK_FATAL"
      sleep 6
    done
    sleep 20   # cool toward nominal between runtimes
  done
  echo "===== matrix done — check each JSON: coldRun=true, initialThermalState=nominal ====="
}

# energy / throttle (sustained, UNPLUGGED — battery must actually drain) is a separate
# protocol; run per runtime when you want the J/token + throttle curve, e.g.:
#   ... --task energy --sustain-seconds 600 --runs 1     (phone off USB, see methodology/energy-ios.md)

collect() {
  local DEST="${DEST:-/tmp/yardstick-collect}"
  mkdir -p "$DEST"
  xcrun devicectl device copy from --device "$UDID" \
    --domain-type appDataContainer --domain-identifier "$APP" \
    --source Documents/results --destination "$DEST" \
    && echo "copied to $DEST — rename into results/raw/ then: python3 scripts/litert_lm_report.py"
}

case "${1:-run}" in
  collect) collect ;;
  *)       run_matrix ;;
esac
