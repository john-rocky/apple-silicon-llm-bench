#!/usr/bin/env python3
"""Camera-VLM throttle chart: sustained FPS + temperature over a 10-minute
continuous camera session, one line per compute placement (ANE vs GPU).

The one-image story: point the same iPhone at the same scene, run Qwen3-VL 2B
on the GPU (MLX) and on the ANE (CoreML), and watch the GPU's FPS collapse as
it heats while the ANE holds.

Reads the camera-VLM result files written by the app (one JSON object each):
    results/raw/*camera-vlm*.json
    results/raw/*camera-vlm*.jsonl

Each carries metrics.fpsOverTime (1 Hz) and metrics.thermalLevelOverTime (1 Hz,
0 nominal · 1 fair · 2 serious · 3 critical) plus a `placement` field.

Run:
    python3 scripts/vlm_throttle_chart.py
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "results", "raw")

# Match the repo chart palette: ANE = amber (CoreML), GPU = violet (MLX).
PLACEMENT_COLOR = {"ane": "#f59e0b", "gpu": "#7c3aed"}
PLACEMENT_LABEL = {"ane": "ANE (CoreML)", "gpu": "GPU (MLX)"}
THERMAL_NAMES = ["nominal", "fair", "serious", "critical"]


def smooth(xs, k=8):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - k)
        window = xs[lo:i + 1]
        out.append(sum(window) / len(window) if window else 0)
    return out


def load(path):
    text = open(path).read().strip()
    return json.loads(text.split("\n")[0])


def main():
    paths = sorted(
        glob.glob(os.path.join(RAW, "*camera-vlm*.json"))
        + glob.glob(os.path.join(RAW, "*camera-vlm*.jsonl"))
    )
    if not paths:
        print("No camera-VLM results found in results/raw/ (*camera-vlm*.json[l]).")
        print("Run a session in the app (Camera tab), export, and import to results/raw/.")
        return

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax2 = ax.twinx()

    drew = False
    for path in paths:
        try:
            d = load(path)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {os.path.basename(path)}: {exc}")
            continue
        m = d.get("metrics", {})
        fps = m.get("fpsOverTime") or []
        therm = m.get("thermalLevelOverTime") or []
        placement = (d.get("placement") or "gpu").lower()
        color = PLACEMENT_COLOR.get(placement, "#64748b")
        label = PLACEMENT_LABEL.get(placement, placement.upper())
        if not fps:
            continue

        ys = smooth([v for v in fps])
        xs = list(range(len(ys)))
        ax.plot(xs, ys, color=color, lw=2.4, label=f"{label} — FPS")
        if ys:
            ax.annotate(f" {ys[-1]:.1f}", (xs[-1], ys[-1]), color=color,
                        fontsize=10, va="center", fontweight="bold")
        # Thermal on the secondary axis as a faint step line.
        if therm:
            tx = list(range(len(therm)))
            ax2.step(tx, therm, color=color, lw=1.2, alpha=0.35, where="post")
        drew = True

    if not drew:
        print("Found result files but none had a fpsOverTime series.")
        return

    ax.set_title("Camera VLM — sustained FPS vs heat (Qwen3-VL 2B, same iPhone, same scene)",
                 fontsize=12.5, fontweight="bold")
    ax.set_xlabel("seconds of continuous camera inference")
    ax.set_ylabel("sustained FPS (5 s rolling)")
    ax.set_ylim(0, None)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)

    ax2.set_ylabel("thermal state (faint)")
    ax2.set_ylim(-0.1, 3.3)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(THERMAL_NAMES)

    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_dir = os.path.join(REPO, "docs", "charts")
    os.makedirs(out_dir, exist_ok=True)
    for out in (os.path.join(out_dir, "vlm-camera-throttle.png"),
                os.path.join(REPO, "results", "vlm-camera-throttle.png")):
        fig.savefig(out, dpi=140)
        print("saved:", out)


if __name__ == "__main__":
    main()
