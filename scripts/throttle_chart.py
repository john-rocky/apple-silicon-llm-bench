#!/usr/bin/env python3
"""Plot sustained decode-throttle curves (tok/s over time) for the iPhone runs.

Reads the energy-task JSONLs (which carry metrics.decodeRateRollingWindow,
~1 sample/sec) and renders one line per runtime to results/iphone17pro-throttle.png.

Run with matplotlib available, e.g.:
    DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib python3 scripts/throttle_chart.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_RAW = "results/raw"
SERIES = [
    ("CoreML / ANE",     f"{REPO_RAW}/iphone17pro-coreml-llm-gemma-4-e2b-energy-tg128.jsonl", "#2ca02c"),
    ("MLX / GPU",        f"{REPO_RAW}/iphone17pro-mlx-swift-gemma-4-e2b-energy-tg128.jsonl",  "#d62728"),
    ("LiteRT-LM / GPU",  f"{REPO_RAW}/iphone17pro-litert-lm-gemma-4-e2b-energy-tg128.jsonl",  "#ff7f0e"),
]


def smooth(xs, k=8):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - k)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def main():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label, path, color in SERIES:
        d = json.loads(open(path).read().strip().split("\n")[0])
        rw = [x for x in d["metrics"].get("decodeRateRollingWindow", []) if x > 0]
        ys = smooth(rw)
        xs = list(range(len(ys)))
        ax.plot(xs, ys, label=label, color=color, lw=2.2)
        ax.annotate(f" {ys[-1]:.0f}", (xs[-1], ys[-1]), color=color,
                    fontsize=10, va="center", fontweight="bold")

    ax.set_title("Sustained decode throttling — iPhone 17 Pro, Gemma 4 E2B (4-bit)",
                 fontsize=12.5)
    ax.set_xlabel("seconds of continuous generation")
    ax.set_ylabel("decode tok/s")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, None)
    ax.margins(x=0.02)
    fig.tight_layout()
    out = "results/iphone17pro-throttle.png"
    fig.savefig(out, dpi=140)
    print("saved:", out)


if __name__ == "__main__":
    main()
