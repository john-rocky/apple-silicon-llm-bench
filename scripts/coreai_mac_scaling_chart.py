#!/usr/bin/env python3
"""Core AI GPU vs MLX, M4 Max, decode tok/s at two model sizes — the gap shrinks
with scale. Matched params (512-token prompt, 512 gen, greedy, 4-bit, warm).
Core AI via Apple's `llm-benchmark`; MLX via `mlx_lm`. Outputs
docs/charts/mac_coreai_scaling.png.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.3,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlepad": 10,
    "figure.dpi": 140, "savefig.dpi": 140, "savefig.bbox": "tight",
})

# Measured on M4 Max, matched (512/512/greedy/warm), 4-bit. decode tok/s.
SIZES = ["Qwen3-0.6B", "Qwen3-8B"]
CORE_AI = [1120.8, 94.3]
MLX = [454.6, 89.7]
RATIO = [c / m for c, m in zip(CORE_AI, MLX)]

x = np.arange(len(SIZES)); w = 0.34
fig, ax = plt.subplots(figsize=(8.0, 4.6))
b1 = ax.bar(x - w / 2, CORE_AI, w, label="Core AI · GPU (pipelined)", color="#e11d48", edgecolor="white")
b2 = ax.bar(x + w / 2, MLX, w, label="MLX · GPU", color="#7c3aed", edgecolor="white")
ax.set_yscale("log")
ax.set_ylim(40, 1900)
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() * 1.04,
                f"{r.get_height():.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
for xi, ratio, ca in zip(x, RATIO, CORE_AI):
    ax.text(xi, ca * 1.55, f"Core AI {ratio:.2f}×", ha="center", fontsize=11,
            fontweight="bold", color="#7f1d1d",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="none"))
ax.set_xticks(x); ax.set_xticklabels(SIZES, fontsize=11)
ax.set_ylabel("decode tok/s  (log scale; ↑ better)", fontsize=10)
ax.set_title("Core AI's GPU lead over MLX shrinks with model size", loc="left")
ax.legend(loc="center right", fontsize=9.5, frameon=True, framealpha=0.9)
fig.suptitle("Apple Core AI vs MLX — M4 Max, 4-bit, matched (512/512, greedy, warm)",
             fontsize=13.5, fontweight="bold", y=1.02)
fig.text(0.5, -0.03, "github.com/john-rocky/apple-silicon-llm-bench  ·  Core AI: llm-benchmark  ·  MLX: mlx_lm",
         ha="center", fontsize=8.5, color="#666")
out = OUT / "mac_coreai_scaling.png"
plt.savefig(out)
plt.close(fig)
print("wrote", out)
