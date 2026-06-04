#!/usr/bin/env python3
"""Bar chart of package power at full decode, per compute unit.

These are whole-package watts measured on Mac (M4 Max) with `powermetrics`
(scripts/measure_energy.py), same model as the iPhone throttle test
(Gemma 4 E2B, 4-bit). iOS exposes no per-subsystem power, so the watts that
*explain* the iPhone throttling are measured here, where powermetrics is
available. The ANE path draws ~half the GPU path — which is why, on the
thermally-constrained iPhone, the GPU runtimes throttle and the ANE doesn't.

    DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib python3 scripts/power_chart.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, avg package watts at full decode, color) — from the README energy table
BARS = [
    ("CoreML / ANE",   12.7, "#2ca02c"),
    ("MLX / GPU",      24.7, "#d62728"),
    ("llama.cpp / GPU", 24.5, "#ff7f0e"),
]


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = [b[0] for b in BARS]
    vals = [b[1] for b in BARS]
    colors = [b[2] for b in BARS]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.1f} W", (bar.get_x() + bar.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Package power at full decode — Apple Silicon (M4 Max, powermetrics)\n"
                 "Gemma 4 E2B (4-bit) — the ANE draws ~half the GPU",
                 fontsize=11.5)
    ax.set_ylabel("avg package power (W)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = "results/apple-silicon-power.png"
    fig.savefig(out, dpi=140)
    print("saved:", out)


if __name__ == "__main__":
    main()
