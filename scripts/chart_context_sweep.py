#!/usr/bin/env python3
"""Context-length sweep + prefill charts — Qwen3-4B iso-int4, iPhone 17 Pro.

  1) decode-vs-context: ANE's short-context lead vanishes at depth (crossover).
  2) prefill tok/s + TTFT: ANE > MLX > GPU >> LiteRT (LiteRT's 25 s prefill wall).

Medians from results/raw/.../context-sweep-qwen3-4b.jsonl (2 cold/cell).
Run: python3 scripts/chart_context_sweep.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
CH = REPO / "docs" / "charts"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})
CORE_AI, GPU_C, MLX, LITERT = "#E8843C", "#F0A868", "#4C78C8", "#2E8B57"

CTX = [19, 666, 2681, 3977]
DEC = {  # decode tok/s
    "Core AI ANE": ([27.4, 26.3, 11.6, 10.5], CORE_AI),
    "Core AI GPU": ([17.0, 16.1, 13.0, 11.9], GPU_C),
    "MLX":         ([20.3, 18.0, 14.6, None], MLX),
    "LiteRT-LM":   ([15.3, 15.5, 13.7, None], LITERT),
}

# ----------------------------------------------------- 1) decode vs context
def chart_decode():
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for lbl, (ys, col) in DEC.items():
        xs = [c for c, y in zip(CTX, ys) if y is not None]
        vs = [y for y in ys if y is not None]
        ax.plot(xs, vs, "-o", color=col, lw=2.4, ms=7, label=lbl)
        if ys[-1] is None:  # mark the failure at the 4096 ceiling
            ax.plot(2681, ys[2], "o", ms=7, color=col)
    ax.annotate("ANE: fastest short,\nslowest at depth", xy=(2681, 11.6), xytext=(1250, 18.5),
                fontsize=9.5, color=CORE_AI, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=CORE_AI, lw=1.4))
    ax.annotate("MLX / LiteRT fail\nnear the 4096 ctx ceiling", xy=(3977, 11), xytext=(2650, 4.5),
                fontsize=8.6, color="#777", ha="left",
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.2))
    ax.set_xlabel("context length (prompt tokens)"); ax.set_ylabel("decode tok/s (higher = faster)")
    ax.set_ylim(0, 30); ax.set_xlim(-100, 4200)
    fig.suptitle("Decode-at-depth — the ANE's short-context lead vanishes by ~2K", fontsize=14, fontweight="bold", y=1.02)
    ax.set_title("Qwen3-4B (iso-int4) · iPhone 17 Pro · short reply after a prompt of the given length", fontsize=10.5, fontweight="normal", color="#333", pad=10)
    ax.legend(loc="upper right", frameon=False)
    fig.text(0.5, -0.04,
             "ANE is fastest at short context (27) but its static-bucket KV attention degrades steeply — by ~2.7K it is "
             "LAST (11.6), behind MLX 14.6 / LiteRT 13.7 / GPU 13.0. The other runtimes degrade gently. So \"ANE is "
             "fastest\" holds only for short prompts; for long context it inverts.",
             ha="center", fontsize=8.6, color="#444", wrap=True)
    out = CH / "iphone_context_sweep_decode.png"; fig.savefig(out); plt.close(fig); print("wrote:", out)

# ----------------------------------------------------- 2) prefill + TTFT
def chart_prefill():
    rows = [  # label, prefill tok/s @2.7k, TTFT s, color
        ("Core AI\nANE", 671, 4.0, CORE_AI),
        ("MLX", 514, 5.2, MLX),
        ("Core AI\nGPU", 359, 7.6, GPU_C),
        ("LiteRT-LM", 108, 24.9, LITERT),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for i, (lbl, pf, ttft, col) in enumerate(rows):
        ax.bar(i, pf, color=col, edgecolor="white")
        ax.text(i, pf + 12, f"{pf}", ha="center", fontweight="bold", fontsize=11)
        if pf > 120:  # in-bar TTFT only for the tall bars; LiteRT's is called out by the annotation
            ax.text(i, pf / 2, f"TTFT\n{ttft:.1f}s", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    ax.annotate("25 s before the first token", xy=(3, 108), xytext=(1.7, 430),
                fontsize=9.5, color="#c0392b", fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4))
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("prefill tok/s @ 2.7K context (higher = faster)"); ax.set_ylim(0, 760)
    fig.suptitle("Prefill — LiteRT-LM's 25-second wall", fontsize=14, fontweight="bold", y=1.02)
    ax.set_title("Qwen3-4B (iso-int4) · iPhone 17 Pro · 2,681-token prompt · prefill tok/s = prompt ÷ TTFT", fontsize=10.5, fontweight="normal", color="#333", pad=10)
    fig.text(0.5, -0.05,
             "ANE prefills fastest (671 tok/s); LiteRT-LM is ~5–6× slower (≈108 tok/s) → 25 s to first token at 2.7K "
             "context vs 4–8 s for the others. LiteRT's decode-at-depth is fine (flat ~14); its on-device weakness is "
             "prefill. (LiteRT reports no prompt-token count, so its rate is derived from TTFT.)",
             ha="center", fontsize=8.6, color="#444", wrap=True)
    out = CH / "iphone_context_sweep_prefill.png"; fig.savefig(out); plt.close(fig); print("wrote:", out)

chart_decode()
chart_prefill()
