#!/usr/bin/env bash
# ============================================================================
# full_matrix.sh — one-shot orchestration for the LiteRT-LM cross-runtime bench
# across BOTH iPhone (devicectl) and Mac (yardstick CLI), all models, all tasks.
#
# Designed to be run in phases once everything is staged:
#   1) build       — build the macOS yardstick CLI (SwiftPM)         [Mac]
#   2) ⌘R          — rebuild+install the iOS app in Release (manual; new catalog rows)
#   3) prefetch    — download every model on the Mac + side-load the iPhone-side ones
#   4) iphone      — run all plugged iPhone jobs (short-chat + long-context, n=3 cold)
#   5) mac         — run all Mac jobs (litert + mlx; short-chat + long-context + sustained)
#   6) iphone-energy — run the UNPLUGGED iPhone energy jobs (separate; phone off USB)
#   7) collect     — pull iPhone JSONL into results/raw/
#   8) report      — regenerate docs/litert-lm/ + the macOS desktop numbers
#
# Usage:
#   UDID=<udid> ./scripts/full_matrix.sh build|prefetch|iphone|mac|iphone-energy|collect|report
#   ./scripts/full_matrix.sh all-mac        # build + mac + report (no iPhone)
# Find UDID: xcrun devicectl list devices
#
# NOTHING here runs automatically — invoke a phase explicitly. Phases are
# idempotent (re-running re-does that phase). See docs/litert-lm/RUNBOOK.md.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
APP=com.iosllmbenchmark.benchmarkapp
YS="$REPO/.build/release/yardstick"
RAW="$REPO/results/raw"
SIDELOAD="${SIDELOAD:-/tmp/sideload}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-1500}"   # generous: first run of each model may download
export HF_HUB_DISABLE_TELEMETRY=1

# ---- THE MATRIX ------------------------------------------------------------
# Columns: runtime | catalog-model-id | hf-repo | file-glob (for side-load; "*" = whole repo)
# device fit: 8B/14B are Mac-tier (phones may jetsam ~>3 GB — that itself is a result, rule 4).
# Runtime tokens: litert-lm | mlx-swift | llama-cpp | coreml-llm.
QWEN_LITERT=(
  "litert-lm|litert-community/Qwen3-0.6B|litert-community/Qwen3-0.6B|qwen3_0_6b_mixed_int4.litertlm"
  "litert-lm|litert-community/Qwen3-4B|litert-community/Qwen3-4B|qwen3_4b_mixed_int4.litertlm"
  "litert-lm|litert-community/Qwen3-8B|litert-community/Qwen3-8B|qwen3_8b_mixed_int4.litertlm"
)
QWEN_MLX=(
  "mlx-swift|mlx-community/Qwen3-0.6B-4bit|mlx-community/Qwen3-0.6B-4bit|*"
  "mlx-swift|mlx-community/Qwen3-4B-4bit|mlx-community/Qwen3-4B-4bit|*"
  "mlx-swift|mlx-community/Qwen3-8B-4bit|mlx-community/Qwen3-8B-4bit|*"
)
QWEN_LLAMA=(
  "llama-cpp|unsloth/Qwen3-4B-GGUF/Q4_K_M|unsloth/Qwen3-4B-GGUF|Qwen3-4B-Q4_K_M.gguf"
  "llama-cpp|unsloth/Qwen3-8B-GGUF/Q4_K_M|unsloth/Qwen3-8B-GGUF|Qwen3-8B-Q4_K_M.gguf"
)
GEMMA=(   # E2B + E4B (E4B ~3.75 GB 4-bit — iPhone borderline; attempt = recorded)
  "litert-lm|litert-community/gemma-4-E2B-it-litert-lm|litert-community/gemma-4-E2B-it-litert-lm|gemma-4-E2B-it.litertlm"
  "mlx-swift|mlx-community/gemma-4-e2b-it-4bit|mlx-community/gemma-4-e2b-it-4bit|*"
  "llama-cpp|unsloth/gemma-4-E2B-it-GGUF/Q4_K_M|unsloth/gemma-4-E2B-it-GGUF|gemma-4-E2B-it-Q4_K_M.gguf"
  "litert-lm|litert-community/gemma-4-E4B-it-litert-lm|litert-community/gemma-4-E4B-it-litert-lm|gemma-4-E4B-it.litertlm"
  "mlx-swift|mlx-community/gemma-4-e4b-it-4bit|mlx-community/gemma-4-e4b-it-4bit|*"
  "llama-cpp|unsloth/gemma-4-E4B-it-GGUF/Q4_K_M|unsloth/gemma-4-E4B-it-GGUF|gemma-4-E4B-it-Q4_K_M.gguf"
)
# Mac-tier big models (≫ iOS per-app memory ceiling): litert + mlx only.
MAC_BIG=(
  "litert-lm|litert-community/Qwen3-14B|litert-community/Qwen3-14B|qwen3_14b_mixed_int4.litertlm"
  "mlx-swift|mlx-community/Qwen3-14B-4bit|mlx-community/Qwen3-14B-4bit|*"
  "litert-lm|litert-community/gemma-4-12B-it-litert-lm|litert-community/gemma-4-12B-it-litert-lm|gemma-4-12B-it.litertlm"
  "mlx-swift|mlx-community/gemma-4-12b-it-4bit|mlx-community/gemma-4-12b-it-4bit|*"
)
# coreml gemma is side-loaded from the local prebuilt bundle (no HF), see prefetch().
COREML_LOCAL="$HOME/Documents/Models/gemma4-e2b"

# iPhone runs the small/mid set (0.6/4B everywhere; 8B + E4B attempted, may jetsam = recorded).
IPHONE_JOBS=( "${QWEN_LITERT[@]}" "${QWEN_MLX[@]}" "${QWEN_LLAMA[@]}" "${GEMMA[@]}" )
# Mac runs everything + the big models; mac() filters to litert+mlx (CLI ships only those).
MAC_JOBS=( "${IPHONE_JOBS[@]}" "${MAC_BIG[@]}" )

# model-id -> short file token (qwen3-0.6b etc.) used in results/raw filenames
model_token() {
  case "$1" in
    *Qwen3-14B*|*Qwen3-14b*)   echo "qwen3-14b" ;;
    *Qwen3-0.6B*|*Qwen3-0.6b*) echo "qwen3-0.6b" ;;
    *Qwen3-4B*|*Qwen3-4b*)     echo "qwen3-4b"  ;;
    *Qwen3-8B*|*Qwen3-8b*)     echo "qwen3-8b"  ;;
    *gemma-4-E4B*|*gemma-4-e4b*) echo "gemma-4-e4b" ;;
    *gemma-4-12B*|*gemma-4-12b*) echo "gemma-4-12b" ;;
    *gemma-4-E2B*|*gemma-4-e2b*) echo "gemma-4-e2b" ;;
    *) echo "$(basename "$1" | tr 'A-Z' 'a-z')" ;;
  esac
}
rt_token() { case "$1" in mlx-swift) echo mlx ;; *) echo "$1" ;; esac; }

# ---- phases ----------------------------------------------------------------
build() {
  echo ">> building macOS yardstick CLI (SwiftPM, Release)"
  GIT_LFS_SKIP_SMUDGE=1 swift build -c release --product yardstick && echo "  -> $YS"
}

prefetch() {   # download on Mac + side-load iPhone-side models
  : "${UDID:?set UDID=<device-udid>}"
  mkdir -p "$SIDELOAD"
  echo ">> prefetch: downloading all models on the Mac (HF), then side-loading to $UDID"
  for job in "${IPHONE_JOBS[@]}"; do
    IFS='|' read -r rt mid repo glob <<<"$job"
    local dest_local="$SIDELOAD/$(echo "$repo" | tr '/' '__')"
    echo "-- $rt $repo ($glob)"
    if [ "$glob" = "*" ]; then hf download "$repo" --local-dir "$dest_local" >/dev/null 2>&1
    else hf download "$repo" --include "$glob" --local-dir "$dest_local" >/dev/null 2>&1; fi
    case "$rt" in
      llama-cpp)
        push_dir "$dest_local" "Documents/models/llama.cpp/$(echo "$repo" | tr '/' '__')" ;;
      litert-lm)
        push_dir "$dest_local" "Documents/models/litert-lm/$(echo "$repo" | tr '/' '__')" ;;
      mlx-swift)
        seed_mlx_hub "$repo" ;;   # blobs+refs into Library/Caches HF hub cache
    esac
  done
  # coreml gemma: push the local prebuilt bundle (no HF)
  [ -d "$COREML_LOCAL" ] && push_dir "$COREML_LOCAL" "Documents/Models/gemma4-e2b" \
    || echo "  (coreml gemma bundle $COREML_LOCAL not found — coreml gemma will be skipped)"
  echo ">> prefetch done"
}

push_dir() {   # $1 local dir -> $2 device container path
  echo "   push $1 -> $2"
  xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
    --domain-identifier "$APP" --source "$1" --destination "$2" 2>&1 | grep -iE "File on Device|error" | tail -1
}
seed_mlx_hub() {   # $1 hf-repo -> seed blobs+refs (HubClient rebuilds snapshots/)
  local repo="$1" hub="$HOME/.cache/huggingface/hub/models--$(echo "$1" | tr '/' '--')"
  [ -d "$hub" ] || hf download "$repo" >/dev/null 2>&1
  local stage="$SIDELOAD/mlxhub-$(echo "$1" | tr '/' '__')/models--$(echo "$1" | tr '/' '--')"
  rm -rf "$stage"; mkdir -p "$stage/blobs" "$stage/refs"
  ln "$hub"/blobs/* "$stage/blobs/" 2>/dev/null || cp "$hub"/blobs/* "$stage/blobs/"
  cp "$hub"/refs/main "$stage/refs/main"
  push_dir "$stage/models--$(echo "$1" | tr '/' '--')" \
    "Library/Caches/huggingface/hub/models--$(echo "$1" | tr '/' '--')"
}

# Plugged tasks (no battery needed): "task-id:runs:timeout". short-chat = tight median;
# long-context sweep (2K/8K/32K) = prefill scaling + decode-at-depth; sustained-generation
# = decode-degradation over a 512-token continuous decode (NOT the hanging sustainSeconds path).
PLUGGED_TASKS=( "short-chat:3:360" "long-context:2:600" "long-context-8k:2:900" \
                "long-context-32k:1:1500" "sustained-generation:1:600" )

iphone() {   # plugged jobs: short-chat + long-context sweep + sustained-generation, cold
  : "${UDID:?set UDID=<device-udid>}"
  for spec in "${PLUGGED_TASKS[@]}"; do
    IFS=':' read -r task n to <<<"$spec"
    for job in "${IPHONE_JOBS[@]}"; do
      IFS='|' read -r rt mid repo glob <<<"$job"
      echo "########## iphone $rt $mid ($task, n=$n) ##########"
      for run in $(seq 1 "$n"); do
        timeout "$to" xcrun devicectl device process launch --console --terminate-existing \
          --device "$UDID" "$APP" -- --yardstick-autorun --runtime "$rt" --model-id "$mid" \
          --task "$task" --runs 1 2>&1 | grep -iE "YARDSTICK_RUN_OK|YARDSTICK_RUN_FAIL|signal 1[15]|no such" | tail -2
        sleep 6
      done
      sleep 25   # cool toward nominal between models
    done
  done
  echo ">> iphone plugged matrix done — verify coldRun=true / initialThermalState=nominal, then: collect"
}

iphone_energy() {   # UNPLUGGED — battery must drain; run with the phone OFF USB
  : "${UDID:?set UDID=<device-udid>}"
  echo ">> iphone-energy: phone must be UNPLUGGED. litert energy is expected to HANG (0.13.x bug) — recorded, rule 4."
  for job in "${IPHONE_JOBS[@]}"; do
    IFS='|' read -r rt mid repo glob <<<"$job"
    [ "$rt" = "litert-lm" ] && { echo "## skip litert energy ($mid) — known sustained hang (DEADLINE_EXCEEDED)"; continue; }
    echo "########## iphone-energy $rt $mid ##########"
    timeout 900 xcrun devicectl device process launch --console --terminate-existing \
      --device "$UDID" "$APP" -- --yardstick-autorun --runtime "$rt" --model-id "$mid" \
      --task energy --sustain-seconds 600 --runs 1 2>&1 | grep -iE "YARDSTICK_RUN_OK|FAIL|signal" | tail -2
    sleep 60
  done
}

mac() {   # litert + mlx via the CLI; short-chat + long-context sweep + sustained-generation
  [ -x "$YS" ] || { echo "CLI missing — run: $0 build"; return 1; }
  for spec in "${PLUGGED_TASKS[@]}"; do
    IFS=':' read -r task n to <<<"$spec"
    for job in "${MAC_JOBS[@]}"; do
      IFS='|' read -r rt mid repo glob <<<"$job"
      [[ "$rt" != "litert-lm" && "$rt" != "mlx-swift" ]] && continue   # CLI ships litert+mlx only
      local out="m4max-$(rt_token "$rt")-$(model_token "$mid")-${task}"
      echo "########## mac $rt $mid ($task, n=$n) ##########"
      for run in $(seq 1 "$n"); do
        local f="$RAW/${out}-run${run}.jsonl"; rm -f "$f"
        GIT_LFS_SKIP_SMUDGE=1 timeout "$to" "$YS" run --task "$task" \
          --runtime "$rt" --model "$mid" --output "$f" 2>&1 | grep -iE "decode=|TTFT=|FAILED|^error" | tail -2
      done
    done
  done
  echo ">> mac matrix done (litert is CPU-only on macOS — see MACOS_DESKTOP.md)"
}

collect() {
  : "${UDID:?set UDID=<device-udid>}"
  local dest="${DEST:-/tmp/yardstick-collect}"; mkdir -p "$dest"
  xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
    --domain-identifier "$APP" --source Documents/results --destination "$dest" \
    && echo ">> collected to $dest — rename into results/raw/<device>-<rt>-<model>-<task>-runN.jsonl, then: $0 report"
}

report() {
  python3 scripts/litert_lm_report.py
  echo ">> regenerated docs/litert-lm/. Update MACOS_DESKTOP.md from the m4max-* files (hand-authored)."
}

case "${1:-help}" in
  build) build ;;
  prefetch) prefetch ;;
  iphone) iphone ;;
  iphone-energy) iphone_energy ;;
  mac) mac ;;
  collect) collect ;;
  report) report ;;
  all-mac) build && mac && report ;;
  *) sed -n '2,40p' "$0" ;;
esac
