#!/usr/bin/env bash
# Fetch the binary + cloned dependencies the BenchmarkApp project needs:
#
#   1. Vendored/llama.xcframework  (llama.cpp Metal build, ~168 MB)
#   2. Vendored/Anemll             (cloned source for AnemllCore SwiftPM target)
#   3. Vendored/LiteRT-LM          (cloned source for the LiteRTLM SwiftPM target)
#   4. Vendored/CoreML-LLM         (cloned source for the CoreMLLLM SwiftPM target)
#   5. Vendored/coreai-models      (cloned source for the CoreAILM SwiftPM target)
#
# Optional: if `xcodegen` is installed and you've edited project.yml, this
# script also regenerates BenchmarkApp.xcodeproj. Most users do NOT need
# xcodegen — the .xcodeproj is committed.
set -euo pipefail

cd "$(dirname "$0")/.."

LLAMA_TAG="${LLAMA_TAG:-b8999}"
LLAMA_ZIP_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-xcframework.zip"
VENDORED_DIR="Vendored"
LLAMA_FRAMEWORK="${VENDORED_DIR}/llama.xcframework"
ANEMLL_DIR="${VENDORED_DIR}/Anemll"
ANEMLL_PKG="${ANEMLL_DIR}/anemll-swift-cli/Package.swift"
LITERTLM_DIR="${VENDORED_DIR}/LiteRT-LM"
# Default = the newest capture pin in environment.lock.json, so a fresh clone
# vendors the engine behind the published rows (the old v0.13.1 default handed
# fresh users an engine three releases behind the leaderboard; 2026-08-26
# rehearsal). Bump this WITH the lock, per release, never casually.
LITERTLM_TAG="${LITERTLM_TAG:-v0.16.0}"

mkdir -p "${VENDORED_DIR}"

# 1. llama.xcframework
if [ ! -d "${LLAMA_FRAMEWORK}" ]; then
    echo "Downloading llama.xcframework (${LLAMA_TAG}) …"
    TMP_ZIP="$(mktemp -t llama-xcf).zip"
    trap 'rm -f "${TMP_ZIP}"' EXIT
    curl -L --fail -o "${TMP_ZIP}" "${LLAMA_ZIP_URL}"
    UNPACK_DIR="$(mktemp -d -t llama-unpack)"
    unzip -q "${TMP_ZIP}" -d "${UNPACK_DIR}"
    if [ -d "${UNPACK_DIR}/llama.xcframework" ]; then
        mv "${UNPACK_DIR}/llama.xcframework" "${LLAMA_FRAMEWORK}"
    elif [ -d "${UNPACK_DIR}/build-apple/llama.xcframework" ]; then
        mv "${UNPACK_DIR}/build-apple/llama.xcframework" "${LLAMA_FRAMEWORK}"
    else
        echo "ERROR: Could not locate llama.xcframework inside the release zip." >&2
        find "${UNPACK_DIR}" -maxdepth 3 -name "llama.xcframework" -print
        exit 1
    fi
    rm -rf "${UNPACK_DIR}"
    # Sidecar tag: the xcframework carries no version marker inside, and the tag is the
    # only fact stamp_engine_pins.sh can put in a row's engineVersion without guessing.
    echo "${LLAMA_TAG}" > "${LLAMA_FRAMEWORK}.tag"
    echo "  -> ${LLAMA_FRAMEWORK}"
else
    echo "${LLAMA_FRAMEWORK} already present."
fi

# 2. Anemll (SwiftPM Package.swift lives in a subdirectory; SPM cannot resolve from a remote URL).
if [ ! -d "${ANEMLL_DIR}" ]; then
    echo "Cloning Anemll …"
    git clone --depth 1 https://github.com/Anemll/Anemll.git "${ANEMLL_DIR}"
else
    echo "${ANEMLL_DIR} already present."
fi

# Patch Anemll's swift-transformers pin from branch:main to a tagged 1.x range.
# Upstream branch:main HEAD removed the Tokenizers product (now target-only),
# which breaks john-rocky/CoreML-LLM. Pinning to 1.0.0..<2.0.0 keeps both happy.
if grep -q 'swift-transformers".*branch: "main"' "${ANEMLL_PKG}"; then
    echo "Patching Anemll Package.swift to pin swift-transformers 1.x …"
    sed -i '' 's|.package(url: "https://github.com/huggingface/swift-transformers", branch: "main").*|.package(url: "https://github.com/huggingface/swift-transformers", "1.0.0"..<"2.0.0"),|' "${ANEMLL_PKG}"
fi

# 3. LiteRT-LM (google-ai-edge). Vendored as a *local* SwiftPM package: the released
# package declares `.unsafeFlags(["-Xlinker","-all_load"])`, and SwiftPM rejects unsafe
# flags from versioned/remote products — local path packages are allowed. GIT_LFS_SKIP_SMUDGE=1
# skips a broken Android LFS object upstream; the iOS/macOS xcframework is a remote
# binaryTarget (release URL), so it's fetched by SwiftPM, not git-LFS.
if [ ! -d "${LITERTLM_DIR}" ]; then
    echo "Cloning LiteRT-LM (${LITERTLM_TAG}) …"
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch "${LITERTLM_TAG}" \
        https://github.com/google-ai-edge/LiteRT-LM.git "${LITERTLM_DIR}"
else
    echo "${LITERTLM_DIR} already present."
fi
# Local API patch (committed: patches/litert-lm-swift-benchmark-maxnumtokens.patch):
# upstream benchmark() computes maxNumTokens internally (prefill+decode+32 = ctx
# 1056 at the defaults) and cannot express e.g. ctx 2048; the harness passes it
# explicitly (MediaPipeRuntime native-benchmark cells). This used to live ONLY
# inside a working Vendored/ dir — a clean clone could not compile the yardstick
# (found 2026-08-26, fresh-clone rehearsal stumble #9). Guarded = idempotent; if
# a future tag ships the parameter upstream, the grep sees it and skips.
if ! grep -q "maxNumTokens: Int? = nil" "${LITERTLM_DIR}/swift/Benchmark.swift"; then
    echo "Applying litert-lm-swift-benchmark-maxnumtokens.patch …"
    git -C "${LITERTLM_DIR}" apply "$(pwd)/patches/litert-lm-swift-benchmark-maxnumtokens.patch" \
        && grep -q "maxNumTokens: Int? = nil" "${LITERTLM_DIR}/swift/Benchmark.swift" \
        || { echo "ERROR: the maxNumTokens patch did not apply. The patches target LiteRT-LM ${LITERTLM_TAG}." >&2; exit 1; }
fi
# Second local patch: MediaPipeRuntime's --litert-fp32-act reads
# ExperimentalFlags.forceFloat32Activations, which upstream does not have, so a clean
# clone cannot compile the yardstick without it. Guarded = idempotent, as above.
if ! grep -q "forceFloat32Activations" "${LITERTLM_DIR}/swift/ExperimentalFlags.swift"; then
    echo "Applying 0002-experimental-flags-force-fp32-activations.patch …"
    git -C "${LITERTLM_DIR}" apply "$(pwd)/Vendored-patches/0002-experimental-flags-force-fp32-activations.patch" \
        && grep -q "forceFloat32Activations" "${LITERTLM_DIR}/swift/ExperimentalFlags.swift" \
        || { echo "ERROR: patch 0002 did not apply. The patches target LiteRT-LM ${LITERTLM_TAG}." >&2; exit 1; }
fi

# 4. CoreML-LLM (john-rocky). Local SwiftPM package (CoreMLLLM product) for the CoreML/ANE
#    runtime. Tracks the latest default branch; pin a commit if you need exact reproduction.
COREMLLLM_DIR="${VENDORED_DIR}/CoreML-LLM"
if [ ! -d "${COREMLLLM_DIR}" ]; then
    echo "Cloning CoreML-LLM …"
    git clone --depth 1 https://github.com/john-rocky/CoreML-LLM.git "${COREMLLLM_DIR}"
else
    echo "${COREMLLLM_DIR} already present."
fi

# 5. coreai-models (Apple Core AI, CoreAILM product). Builds its grammar bridge from xgrammar source.
#    Required for the project to *compile* (project.yml links it) even though Core AI *runs*
#    additionally need a side-loaded .aimodel in Documents/CoreAIModels/<id>/.
COREAI_DIR="${VENDORED_DIR}/coreai-models"
COREAI_TAG="${COREAI_TAG:-0.2.0}"
# Gemma-4's per-layer-embedding (PLE) models need StaticInputBuffer (EngineOptions.staticInputBuffers)
# to feed their table to the `_tbl` decode graph. That API does NOT exist in Apple's public
# coreai-models (checked: absent from both 0.1.0 and 0.2.0) — it comes from our engine patch,
# patches/coreai-models-938d0b8-gemma4-ple.diff at the repo root. CoreAIRuntime.swift uses it
# only under COREAI_STATIC_INPUTS (off by default), so the public tag builds the app; PLE models
# then report unsupported (methodology/core-ai-arm-provenance.md).
# Prefer a local checkout that carries the PLE patch (what project.yml's comment assumes);
# otherwise clone the public tag.
# Default = a dedicated checkout of 0.2.0 with the PLE patch applied. COREAI_LOCAL overrides it.
COREAI_LOCAL="${COREAI_LOCAL:-$HOME/code/coreai-models-020-bench}"
if [ ! -e "${COREAI_DIR}" ]; then
    if [ -d "${COREAI_LOCAL}" ] && grep -qrs "public struct StaticInputBuffer" "${COREAI_LOCAL}/swift/Sources"; then
        echo "Linking coreai-models -> ${COREAI_LOCAL} (patched: has StaticInputBuffer)"
        ln -s "${COREAI_LOCAL}" "${COREAI_DIR}"
    else
        echo "No patched coreai-models at ${COREAI_LOCAL}: cloning public ${COREAI_TAG}."
        echo "  Core AI builds; Gemma-4 E2B/E4B (PLE) report unsupported without the patched engine."
        git clone --depth 1 --branch "${COREAI_TAG}" https://github.com/apple/coreai-models.git "${COREAI_DIR}"
    fi
else
    echo "${COREAI_DIR} already present."
fi
# Local patch for the public 0.2.0 clone (ios/BenchmarkApp/patches/coreai-models-0.2.0-xcode27.patch):
#   - Xcode 27.0 (27A266a) renamed FoundationModels' LanguageModelCapabilities(capabilities:)
#     to LanguageModelCapabilities(_:), so 0.2.0 no longer compiles. Upstream fixed the call in
#     5ed9981 (#123), which no 0.2.x tag carries (apple/coreai-models#293).
#   - 0.2.0 takes xgrammar from branch main; the patch pins 0.2.9 (main on 2026-11-07), so
#     every fresh clone builds the same grammar library.
# Only a clean checkout of the 0.2.0 tag is patched; a link or a locally edited checkout is
# left alone.
COREAI_LM_SWIFT="${COREAI_DIR}/swift/Sources/CoreAILanguageModels/LanguageModel/CoreAILanguageModel.swift"
if [ ! -L "${COREAI_DIR}" ] && [ "${COREAI_TAG}" = "0.2.0" ] \
    && grep -qs "LanguageModelCapabilities(capabilities:" "${COREAI_LM_SWIFT}" \
    && git -C "${COREAI_DIR}" diff --quiet HEAD; then
    echo "Applying coreai-models-0.2.0-xcode27.patch …"
    git -C "${COREAI_DIR}" apply "$(pwd)/patches/coreai-models-0.2.0-xcode27.patch" \
        && ! grep -q "LanguageModelCapabilities(capabilities:" "${COREAI_LM_SWIFT}" \
        || { echo "ERROR: coreai-models-0.2.0-xcode27.patch did not apply to ${COREAI_DIR}." >&2; exit 1; }
fi

# 6. Cactus engine (cactus-compute/cactus) — build cactus-ios.xcframework from source.
#    The framework bundles cactus_engine.h + a module map, so `import cactus` works
#    directly; CactusRuntime.swift is canImport-guarded and stubs out if this step
#    is skipped (e.g. no cmake). Requires cmake >= 3.10 + Xcode iOS SDK.
CACTUS_DIR="${VENDORED_DIR}/cactus"
CACTUS_FRAMEWORK="${VENDORED_DIR}/cactus-ios.xcframework"
# Pinned: cactus main moves fast and artifacts change under it (their 07-09 CQ4 swap
# is a table-level finding in this repo). This is the commit the published cells used.
CACTUS_COMMIT="${CACTUS_COMMIT:-1ace6d78}"
if [ ! -d "${CACTUS_FRAMEWORK}" ]; then
    if command -v cmake >/dev/null 2>&1; then
        if [ ! -d "${CACTUS_DIR}" ]; then
            echo "Cloning cactus @ ${CACTUS_COMMIT} …"
            git clone https://github.com/cactus-compute/cactus.git "${CACTUS_DIR}"
            if ! (cd "${CACTUS_DIR}" && git checkout --quiet "${CACTUS_COMMIT}"); then
                # This commit is no longer on GitHub. Drop the clone so stamp_engine_pins.sh
                # cannot record another commit as the Cactus version.
                echo "WARNING: cactus ${CACTUS_COMMIT} is no longer on GitHub."
                rm -rf "${CACTUS_DIR}"
            fi
        fi
        if [ -d "${CACTUS_DIR}" ]; then
            echo "Building cactus-ios.xcframework (this compiles the engine twice: device + simulator) …"
        fi
        if [ -d "${CACTUS_DIR}" ] && (cd "${CACTUS_DIR}" && bash apple/build.sh); then
            cp -R "${CACTUS_DIR}/apple/cactus-ios.xcframework" "${CACTUS_FRAMEWORK}"
            # Upstream bundles only cactus_engine.h, whose first include is the C++
            # cactus_graph.h — absent from the framework, so `import cactus` cannot
            # compile as shipped. The FFI declarations never use graph types; drop
            # the include from the bundled header (our copy only).
            for h in "${CACTUS_FRAMEWORK}"/*/cactus.framework/Headers/cactus_engine.h; do
                sed -i '' '/#include "cactus_graph.h"/d' "$h"
            done
            echo "  -> ${CACTUS_FRAMEWORK}"
        else
            echo "WARNING: cactus was not built; the app will report Cactus as unavailable."
        fi
    else
        echo "WARNING: cmake not found; skipping the Cactus arm (CactusRuntime stubs out)."
    fi
else
    echo "${CACTUS_FRAMEWORK} already present."
fi

# 7. Optional: regenerate the Xcode project (only needed if you edit project.yml).
if command -v xcodegen >/dev/null 2>&1; then
    if [ "${REGEN_XCODEPROJ:-0}" = "1" ] || [ ! -d "BenchmarkApp.xcodeproj" ]; then
        echo "Generating BenchmarkApp.xcodeproj …"
        xcodegen generate
    else
        echo "BenchmarkApp.xcodeproj already present (set REGEN_XCODEPROJ=1 to force regen)."
    fi
fi

# Without the Cactus framework the project cannot link, so take it out of the project.
# CactusRuntime.swift is guarded with canImport(cactus), and the app reports Cactus as unavailable.
if [ ! -d "${CACTUS_FRAMEWORK}" ] && grep -q "cactus-ios.xcframework" BenchmarkApp.xcodeproj/project.pbxproj; then
    echo "Removing cactus-ios.xcframework from BenchmarkApp.xcodeproj (it was not built) …"
    sed -i '' '/cactus-ios\.xcframework/d' BenchmarkApp.xcodeproj/project.pbxproj
fi

# 8. Record the observed engine pins (git state of the Vendored/ clones, the CLiteRTLM
#    binaryTarget zip, the llama sidecar tag) into Vendored/engine-pins.json. A build
#    phase re-runs this and stamps the app's Info.plist, so every result row carries
#    engineVersion/engineArtifact (schema v1; continuous-bench condition 3).
./scripts/stamp_engine_pins.sh

cat <<'EOF'

Done. Open the project:
    open BenchmarkApp.xcodeproj

In Xcode:
  1. Set your Apple Developer Team in Signing & Capabilities.
  2. Select your iPhone as the run destination.
  3. ⌘R.

LiteRT-LM (Gemma .litertlm, GPU) is vendored above (Vendored/LiteRT-LM) and wired via
project.yml — no manual Xcode step. Its decode tok/s comes from LiteRT-LM's own counters.

First Xcode build resolves SPM (mlx-swift-lm, swift-huggingface, swift-transformers,
executorch, Anemll, CoreMLLLM). Takes several minutes.
EOF
