#!/usr/bin/env python3
"""Chart: Core AI vs LiteRT-LM vs MLX/CoreML/llama decode on iPhone 17 Pro.

Two panels from results/raw (medians of the 3 cold runs, never hand-typed):
  - Qwen3-0.6B  — Core AI GPU shown cold→warm (Metal kernel cache), ANE cold-consistent.
  - Gemma 4 E2B — the ranking flips: LiteRT-LM leads.
Output: docs/charts/iphone_coreai_vs_litert.png

    python3 scripts/chart_coreai_litert.py
"""
import json, glob, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw"
OUT = REPO / "docs" / "charts" / "iphone_coreai_vs_litert.png"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})

CORE_AI, LITERT, MLX, LLAMA, COREML = "#E8843C", "#2E8B57", "#4C78C8", "#9467BD", "#9AA0A6"

def decodes(pat):
    fs = sorted(glob.glob(str(RAW / pat)))
    return [json.load(open(f))["metrics"]["decodeTokensPerSecond"] for f in fs]

def med(pat):
    v = decodes(pat); return st.median(v) if v else None

# --- Qwen3-0.6B (Core AI GPU: cold = slowest run, warm = median of the rest — kernel cache) ---
gpu = decodes("iphone17pro-core-ai-qwen3-0.6b-gpu-short-chat-run*.jsonl")
gpu_cold = min(gpu); gpu_warm = st.median([x for x in gpu if x != gpu_cold])
q = [  # (label, value, color, cold_for_split_bar)
    ("Core AI\nGPU", gpu_warm, CORE_AI, gpu_cold),
    ("Core AI\nANE", med("iphone17pro-core-ai-qwen3-0.6b-ane-short-chat-run*.jsonl"), CORE_AI, None),
    ("MLX\nGPU", med("iphone17pro-mlx-swift-qwen3-0.6b-short-chat-run*.jsonl"), MLX, None),
    ("LiteRT-LM\nGPU", med("iphone17pro-litert-lm-qwen3-0.6b-short-chat-run*.jsonl"), LITERT, None),
    ("CoreML\nANE", med("iphone17pro-coreml-llm-qwen3-0.6b-short-chat-run*.jsonl"), COREML, None),
]
# --- Gemma 4 E2B ---
gm = [
    ("LiteRT-LM\nGPU", med("iphone17pro-litert-lm-gemma-4-e2b-short-chat-run*.jsonl"), LITERT),
    ("MLX\nGPU", med("iphone17pro-mlx-gemma-4-e2b-short-chat-run*.jsonl"), MLX),
    ("llama.cpp\nGPU", med("iphone17pro-llama-cpp-gemma-4-e2b-short-chat-run*.jsonl"), LLAMA),
]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [5, 3]})

# Panel 1 — Qwen3-0.6B
for i, (lbl, val, col, cold) in enumerate(q):
    if cold is not None:  # cold→warm split bar
        a1.bar(i, cold, color=col, edgecolor="white")
        a1.bar(i, val - cold, bottom=cold, color=col, alpha=0.45, hatch="//", edgecolor="white")
        a1.text(i, val + 4, f"{val:.0f}", ha="center", fontweight="bold")
        a1.text(i, cold - 11, f"cold {cold:.0f}", ha="center", fontsize=8.5, color="white", fontweight="bold")
        a1.annotate("warm", (i, (cold+val)/2), ha="center", va="center", fontsize=8.5, color="#5a3210")
    else:
        a1.bar(i, val, color=col, edgecolor="white")
        a1.text(i, val + 4, f"{val:.0f}", ha="center", fontweight="bold")
a1.set_xticks(range(len(q))); a1.set_xticklabels([x[0] for x in q])
a1.set_ylabel("decode tok/s (higher = faster)"); a1.set_ylim(0, 215)
a1.set_title("Qwen3-0.6B  —  cold: ANE · warm: Core AI GPU", fontsize=12, pad=10)

# Panel 2 — Gemma 4 E2B
for i, (lbl, val, col) in enumerate(gm):
    a2.bar(i, val, color=col, edgecolor="white")
    a2.text(i, val + 1.2, f"{val:.0f}", ha="center", fontweight="bold")
a2.set_xticks(range(len(gm))); a2.set_xticklabels([x[0] for x in gm])
a2.set_ylim(0, 70); a2.set_title("Gemma 4 E2B  —  LiteRT-LM leads", fontsize=12, pad=10)

fig.suptitle("On-device decode — iPhone 17 Pro · one headless harness · greedy · cold · median of 3",
             fontsize=13, fontweight="bold", y=1.02)
fig.text(0.5, -0.04, "Same conditions for every runtime; rates cross-checked against measured inter-token "
         "latency. Quant is each runtime's native format (disclosed per row in the package). "
         "Core AI GPU's Metal kernel cache makes its first run cold (77) and later runs warm (194).",
         ha="center", fontsize=8.5, color="#444", wrap=True)
fig.savefig(OUT)
print("wrote:", OUT)
for lbl, val, *_ in q: print(f"  Qwen3-0.6B {lbl.replace(chr(10),' '):14} {val:.1f}")
print(f"  (Core AI GPU cold={gpu_cold:.1f} warm={gpu_warm:.1f})")
for lbl, val, _ in gm: print(f"  Gemma4-E2B {lbl.replace(chr(10),' '):14} {val:.1f}")
