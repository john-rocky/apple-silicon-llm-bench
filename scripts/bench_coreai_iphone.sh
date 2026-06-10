#!/usr/bin/env bash
#
# bench_coreai_iphone.sh — headless iPhone bench: Core AI vs CoreML-LLM vs MLX
# on Qwen3-0.6B, driven from the Mac via devicectl.
#
# Core AI on iPhone needs AOT-compiled `.aimodelc` (iOS can't JIT the IR), and
# the compute unit is fixed by the EXPORT SHAPE: the static iOS export → ANE
# (static-shape engine), a dynamic export → GPU (pipelined engine). So we ship
# TWO compiled bundles. See methodology/coreai-ios.md.
#
# Requires: iPhone on iOS 27 (udid below); Xcode 27 + coreai-build; the
# coreai-models Swift package symlinked at ios/BenchmarkApp/Vendored/coreai-models.
#
# Usage: scripts/bench_coreai_iphone.sh [udid]
set -euo pipefail

UDID="${1:-00008150-0018713A0207801C}"          # iPhone 17 Pro (iPhone18,1, arch h18p)
ARCH="${COREAI_ARCH:-h18p}"
BUNDLE_ID="com.iosllmbenchmark.benchmarkapp"
TEAM="MFN25KNUGJ"
DEVICE_LABEL="iphone17pro"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROJ="$REPO/ios/BenchmarkApp/BenchmarkApp.xcodeproj"
DD="$HOME/Library/Developer/Xcode/DerivedData/BenchmarkApp-coreai"
APP="$DD/Build/Products/Debug-iphoneos/BenchmarkApp.app"

EXPORTS="$HOME/code/coreai/coreai-models/exports"
MLX_REPO="mlx-community/Qwen3-0.6B-4bit"
MLX_CACHE="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit"
COREML_SRC="$HOME/Documents/Models/qwen3-0.6b/qwen3_0_6b_stateful_chunks"
PULL_DIR="/tmp/coreai_iphone_results"

log() { printf '\n=== %s ===\n' "$*"; }
copy_to() { xcrun devicectl device copy to --device "$UDID" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" --source "$1" --destination "$2"; }

# ---- 1. build + install -----------------------------------------------------
build_install() {
  log "build + install BenchmarkApp -> $UDID"
  xcodebuild -project "$PROJ" -scheme BenchmarkApp -configuration Debug \
    -destination "platform=iOS,id=$UDID" -derivedDataPath "$DD" \
    -allowProvisioningUpdates CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM="$TEAM" build
  xcrun devicectl device install app --device "$UDID" "$APP"
}

# ---- 2. prep the two AOT-compiled Core AI bundles ---------------------------
# Assemble a loadable bundle from a compiled `.aimodelc`: the device-arch file +
# tokenizer + a metadata.json whose assets.main points at the compiled file.
assemble() { # <ir.aimodel> <compute: gpu|neural-engine> <out-bundle-dir> <base>
  local ir="$1" compute="$2" out="$3" base="$4" tmp
  tmp="$(mktemp -d)"
  xcrun coreai-build compile "$ir" --platform iOS --preferred-compute "$compute" \
    --architecture "$ARCH" --output "$tmp"
  rm -rf "$out"; mkdir -p "$out"
  cp -R "$tmp/${base}.${ARCH}.aimodelc" "$out/"
  cp -R "$(dirname "$ir")/../tokenizer" "$out/" 2>/dev/null || cp -R "$(dirname "$(dirname "$ir")")/tokenizer" "$out/"
  python3 - "$(dirname "$ir")/../metadata.json" "$out/metadata.json" "${base}.${ARCH}.aimodelc" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); m["assets"]["main"]=sys.argv[3]
json.dump(m,open(sys.argv[2],"w"),indent=2)
PY
  rm -rf "$tmp"
}

prep_coreai() {
  log "export + AOT-compile Core AI bundles (ANE static, GPU dynamic)"
  ( cd "$HOME/code/coreai/coreai-models"
    [ -d "$EXPORTS/qwen3_0_6b_ios" ]     || uv run coreai.llm.export qwen3-0.6b --platform iOS  --output-name qwen3_0_6b_ios
    [ -d "$EXPORTS/qwen3_0_6b_dynamic" ] || uv run coreai.llm.export qwen3-0.6b --platform macOS --output-name qwen3_0_6b_dynamic )
  assemble "$EXPORTS/qwen3_0_6b_ios/qwen3_0_6b_ios.aimodel"         neural-engine "$EXPORTS/qwen3_0_6b_ane" qwen3_0_6b_ios
  assemble "$EXPORTS/qwen3_0_6b_dynamic/qwen3_0_6b_dynamic.aimodel" gpu           "$EXPORTS/qwen3_0_6b_gpu" qwen3_0_6b_dynamic
}

# ---- 3. side-load -----------------------------------------------------------
sideload() {
  log "side-load Core AI (ANE + GPU) + CoreML + MLX"
  copy_to "$EXPORTS/qwen3_0_6b_ane" "Documents/CoreAIModels/qwen3_0_6b_ane"
  copy_to "$EXPORTS/qwen3_0_6b_gpu" "Documents/CoreAIModels/qwen3_0_6b_gpu"
  copy_to "$COREML_SRC" "Documents/Models/qwen3-0.6b/qwen3_0_6b_stateful_chunks"
  if [ -d "$MLX_CACHE/blobs" ]; then
    local d="Library/Caches/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit"
    copy_to "$MLX_CACHE/blobs" "$d/blobs"; copy_to "$MLX_CACHE/refs" "$d/refs"
  fi
}

# ---- 4. run matrix ----------------------------------------------------------
# DETACHED launch (no --console — it fails to attach from a non-interactive
# shell, CoreDeviceError 10002). --runs 3 = 1 cold + 2 warm in one app session.
run_matrix() {
  local engines=(
    "core-ai core-ai/qwen3-0.6b-ane"
    "core-ai core-ai/qwen3-0.6b-gpu"
    "mlx-swift $MLX_REPO"
    "coreml-llm coreml-llm/qwen3-0.6b"
  )
  for e in "${engines[@]}"; do
    set -- $e
    log "run $1 $2 (--runs 3)"
    xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
      --yardstick-autorun --runtime "$1" --model-id "$2" --task short-chat --runs 3 >/dev/null
    sleep 110
  done
}

# ---- 5. pull + import + render ---------------------------------------------
pull_import_render() {
  log "pull -> import -> render"
  rm -rf "$PULL_DIR"; mkdir -p "$PULL_DIR"
  xcrun devicectl device copy from --device "$UDID" \
    --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
    --source "Documents/results" --destination "$PULL_DIR"
  REPO="$REPO" DEVICE_LABEL="$DEVICE_LABEL" PULL_DIR="$PULL_DIR" python3 - <<'PY'
import json, os, re, glob, pathlib
repo=pathlib.Path(os.environ["REPO"]); dev=os.environ["DEVICE_LABEL"]
raw=repo/"results"/"raw"; raw.mkdir(parents=True, exist_ok=True)
short_rt=lambda r:{"mlx-swift":"mlx"}.get(r,r)
short_m=lambda mid:re.sub(r"[^a-z0-9.\-]+","-",mid.split("/")[-1].lower())
c={}
for f in sorted(glob.glob(os.path.join(os.environ["PULL_DIR"],"**","*.json"),recursive=True)):
    try: d=json.loads(pathlib.Path(f).read_text())
    except Exception as e: print("skip",f,e); continue
    rt=d.get("runtime","?"); mid=(d.get("model") or {}).get("id","?"); task=d.get("task","?")
    k=(rt,mid,task); c[k]=c.get(k,0)+1
    (raw/f"{dev}-{short_rt(rt)}-{short_m(mid)}-{task}-run{c[k]}.jsonl").write_text(json.dumps(d))
print("imported", sum(c.values()), "rows")
PY
  ( cd "$REPO" && python3 scripts/render_results.py ) || true
}

# ---- main -------------------------------------------------------------------
build_install
prep_coreai
sideload
run_matrix
pull_import_render
log "done — see results/raw/$DEVICE_LABEL-* and RESULTS.md"
