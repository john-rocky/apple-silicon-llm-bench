#!/usr/bin/env bash
# Sequentially download + bench the M4-Max-class models.
# Running these in parallel timed out the HF downloader, so this script
# walks them one at a time. ~72 GB of downloads total — leave it running.
set -e

YS=/tmp/yardstick-dd/Build/Products/Release/yardstick
RAW=results/raw

run3() {
    local runtime=$1 model=$2 outname=$3
    echo ""
    echo "=== $outname ==="
    for run in 1 2 3; do
        out="$RAW/m4max-${outname}-short-chat-run${run}.jsonl"
        rm -f "$out"
        $YS run --task short-chat --runtime "$runtime" --model "$model" --output "$out" 2>&1 \
            | grep -E "TTFT|FAILED|error|download progress: (25|50|75|100)%" | tail -3
    done
}

cd "$(dirname "$0")/.."

run3 llama-cpp "unsloth/Qwen3.5-9B-GGUF/Q4_K_M"           "llama-cpp-qwen3.5-9b"
run3 mlx-swift "mlx-community/Qwen3.5-27B-4bit"           "mlx-qwen3.5-27b"
run3 mlx-swift "mlx-community/gemma-4-31b-it-4bit"        "mlx-gemma-4-31b"
run3 mlx-swift "mlx-community/gemma-4-26b-a4b-it-4bit"    "mlx-gemma-4-26b-a4b"
run3 mlx-swift "mlx-community/Qwen3.5-35B-A3B-4bit"       "mlx-qwen3.5-35b-a3b"

echo ""
echo "=== All done — regenerating tables + charts ==="
python3 scripts/render_results.py
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib /opt/homebrew/bin/python3 scripts/generate_charts.py
