#!/usr/bin/env python3
"""
Yardstick — generate the README hero charts from results/raw/*.jsonl.

Outputs to docs/charts/:

    decode_tok_per_s.png     # MLX vs llama.cpp vs CoreML, 5 models, M4 Max
    energy_per_token.png     # 4 backends, Gemma 4 E2B sustained, M4 Max
    itl_jitter.png           # Inter-token p99 ms, 4 backends
    tradeoff.png             # tok/s × J/tok scatter, the Pareto picture

Regenerate after every results/raw/ change:

    python scripts/generate_charts.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw"
OUT = REPO / "docs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# House style — slightly opinionated, but consistent across the chart set.
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
})

PALETTE = {
    "mlx-swift": "#7c3aed",   # MLX = violet
    "llama.cpp": "#0ea5e9",   # llama.cpp = sky
    "coreml-llm": "#f59e0b",  # CoreML/ANE = amber
    "apple-fm":  "#10b981",   # Apple FM = emerald
}

LOGICAL_MODEL_ORDER = [
    "Qwen 2.5 0.5B",
    "Qwen 3.5 0.8B",
    "Qwen 3.5 2B",
    "Gemma 4 E2B",
    "Gemma 4 E4B",
]

LOGICAL_MODEL_MATCH = [
    ("Qwen 2.5 0.5B", ("qwen2.5-0.5b",)),
    ("Qwen 3.5 0.8B", ("qwen3.5-0.8b",)),
    ("Qwen 3.5 2B",   ("qwen3.5-2b",)),
    ("Gemma 4 E2B",   ("gemma-4-e2b", "gemma4-e2b")),
    ("Gemma 4 E4B",   ("gemma-4-e4b", "gemma4-e4b")),
]


def logical_model(model_id: str) -> str | None:
    s = model_id.lower()
    for label, needles in LOGICAL_MODEL_MATCH:
        if any(n in s for n in needles):
            return label
    return None


def load_runs(device: str, task: str | None = None) -> list[dict]:
    out = []
    for p in sorted(RAW.glob(f"{device}-*.jsonl")):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        if task and obj.get("task") != task:
            continue
        out.append(obj)
    return out


def median(values: list[float]) -> float | None:
    xs = [v for v in values if v is not None]
    return statistics.median(xs) if xs else None


# ------------------------------------------------------------------ #
#  Chart 1 — decode tok/s, 3 runtimes × 5 models, M4 Max, short-chat
# ------------------------------------------------------------------ #


def chart_decode_tok_per_s():
    runs = load_runs("m4max", task="short-chat")
    cells: dict[tuple[str, str], list[float]] = {}
    for r in runs:
        lm = logical_model((r.get("model") or {}).get("id", ""))
        if lm is None:
            continue
        rt = r.get("runtime")
        if rt == "apple-fm":  # Apple FM has its own model, not a peer here
            continue
        cells.setdefault((lm, rt), []).append(r["metrics"]["decodeTokensPerSecond"])

    runtimes = ["mlx-swift", "llama.cpp", "coreml-llm"]
    x = list(range(len(LOGICAL_MODEL_ORDER)))
    width = 0.27

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, rt in enumerate(runtimes):
        ys = [median(cells.get((lm, rt), [])) or 0 for lm in LOGICAL_MODEL_ORDER]
        offsets = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(offsets, ys, width, label=rt, color=PALETTE[rt], edgecolor="white", linewidth=0.5)
        for bar, y in zip(bars, ys):
            if y > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, y + 6, f"{y:.0f}",
                        ha="center", va="bottom", fontsize=8, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(LOGICAL_MODEL_ORDER)
    ax.set_ylabel("Decode tok/s (median, n=3)")
    ax.set_title("Decode throughput — M4 Max, short-chat (128 tok)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.text(0.5, -0.04,
             "MLX-Swift wins every cell (1.4×–1.8× over llama.cpp) after early-2026 mlx-swift-lm Qwen + Gemma kernels landed.",
             ha="center", fontsize=8.5, color="#555")
    plt.savefig(OUT / "decode_tok_per_s.png")
    plt.close(fig)
    print(f"wrote {OUT / 'decode_tok_per_s.png'}")


# ------------------------------------------------------------------ #
#  Chart 2 — J/token, 4 runtimes, Gemma 4 E2B sustained, M4 Max
# ------------------------------------------------------------------ #


def chart_energy_per_token():
    runs = load_runs("m4max")
    rows: dict[str, dict] = {}
    for r in runs:
        m = r.get("metrics") or {}
        if m.get("energySource") != "powermetrics":
            continue
        if r.get("task") != "sustained-generation":
            continue
        rt = r.get("runtime")
        rows[rt] = {
            "j_per_tok": m.get("energyJoulesPerToken") or 0,
            "j_total": m.get("energyJoules") or 0,
            "avg_w": m.get("averagePackagePowerW") or 0,
        }

    order = ["apple-fm", "mlx-swift", "llama.cpp", "coreml-llm"]
    labels = [r for r in order if r in rows]
    j_per_tok = [rows[r]["j_per_tok"] for r in labels]

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    colors = [PALETTE[r] for r in labels]
    bars = ax.barh(range(len(labels)), j_per_tok, color=colors, edgecolor="white", linewidth=0.6)
    for bar, v, r in zip(bars, j_per_tok, labels):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f} J/tok  ·  {rows[r]['avg_w']:.1f} W avg",
                va="center", fontsize=9.5, color="#222")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Joules per generated token (lower = better)")
    ax.set_xlim(0, max(j_per_tok) * 1.45)
    ax.set_title("Energy per token — M4 Max, Gemma 4 E2B, sustained-512")
    fig.text(0.5, -0.06,
             "Apple FM ≈ 2× more efficient than the GPU backends; CoreML/ANE's low W is offset by slower decode → 4× the energy.",
             ha="center", fontsize=8.5, color="#555")
    plt.savefig(OUT / "energy_per_token.png")
    plt.close(fig)
    print(f"wrote {OUT / 'energy_per_token.png'}")


# ------------------------------------------------------------------ #
#  Chart 3 — ITL p99, 4 runtimes, Gemma 4 E2B short-chat, M4 Max
# ------------------------------------------------------------------ #


def chart_itl_jitter():
    runs = load_runs("m4max", task="short-chat")
    cells: dict[str, list[float]] = {}
    for r in runs:
        m = r.get("metrics") or {}
        p99 = m.get("interTokenLatencyP99MS")
        if p99 is None:
            continue
        rt = r.get("runtime")
        lm = logical_model((r.get("model") or {}).get("id", ""))
        if rt == "apple-fm" or lm == "Gemma 4 E2B":
            cells.setdefault(rt, []).append(p99)

    order = ["mlx-swift", "llama.cpp", "coreml-llm", "apple-fm"]
    labels = [r for r in order if r in cells]
    ys = [median(cells[r]) for r in labels]

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    colors = [PALETTE[r] for r in labels]
    bars = ax.barh(range(len(labels)), ys, color=colors, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, ys):
        ax.text(v + 3, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f} ms", va="center", fontsize=9.5, color="#222")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Inter-token latency p99 (ms) — lower = smoother stream")
    ax.set_xlim(0, max(ys) * 1.2)
    ax.set_title("Streaming smoothness — M4 Max, Gemma 4 E2B (Apple FM uses own model)")
    fig.text(0.5, -0.05,
             "Apple FM streams in word-sized bursts (200 ms p99) vs MLX/llama.cpp's per-token cadence (5–10 ms). Same avg tok/s, very different chat UX.",
             ha="center", fontsize=8.5, color="#555")
    plt.savefig(OUT / "itl_jitter.png")
    plt.close(fig)
    print(f"wrote {OUT / 'itl_jitter.png'}")


# ------------------------------------------------------------------ #
#  Chart 4 — tok/s vs J/tok scatter (the Pareto picture)
# ------------------------------------------------------------------ #


def chart_tradeoff():
    runs = load_runs("m4max")
    points: dict[str, tuple[float, float]] = {}
    for r in runs:
        m = r.get("metrics") or {}
        if m.get("energySource") != "powermetrics":
            continue
        if r.get("task") != "sustained-generation":
            continue
        rt = r.get("runtime")
        points[rt] = (m["decodeTokensPerSecond"], m["energyJoulesPerToken"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for rt, (tok_s, j_per_tok) in points.items():
        ax.scatter(tok_s, j_per_tok, s=240, color=PALETTE[rt], edgecolor="white", linewidth=1.4, zorder=5)
        ax.annotate(rt,
                    (tok_s, j_per_tok),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=11, fontweight="bold", color="#222")
        ax.annotate(f"{j_per_tok:.2f} J/tok · {tok_s:.0f} tok/s",
                    (tok_s, j_per_tok),
                    textcoords="offset points",
                    xytext=(8, -14),
                    fontsize=8.5, color="#555")

    ax.set_xlabel("Decode throughput (tok/s, sustained-512) — right = faster")
    ax.set_ylabel("Energy per token (J/tok) — down = more efficient")
    ax.set_title("Throughput × Energy — M4 Max, Gemma 4 E2B (Apple FM uses own model)")
    ax.invert_yaxis()
    ax.set_xlim(0, max(p[0] for p in points.values()) * 1.2)
    ax.set_ylim(max(p[1] for p in points.values()) * 1.15, 0)
    fig.text(0.5, -0.02,
             "Apple FM owns the efficiency Pareto frontier; MLX-Swift owns throughput. No runtime is Pareto-dominant.",
             ha="center", fontsize=8.5, color="#555")
    plt.savefig(OUT / "tradeoff.png")
    plt.close(fig)
    print(f"wrote {OUT / 'tradeoff.png'}")


# ------------------------------------------------------------------ #
#  Chart 0 — iPhone 17 Pro headline: decode tok/s + peak memory
# ------------------------------------------------------------------ #

def chart_iphone():
    runs = load_runs("iphone17pro", task="short-chat")
    models = ["Qwen 3.5 2B", "Gemma 4 E2B"]
    # litert-lm first to spotlight the Gemma upset; no LiteRT Qwen bar captured yet
    # (Qwen3-0.6B .litertlm is wired — its device run is pending).
    runtimes = ["litert-lm", "mlx-swift", "llama.cpp", "coreml-llm"]
    rt_label = {"litert-lm": "LiteRT-LM", "mlx-swift": "MLX-Swift",
                "llama.cpp": "llama.cpp", "coreml-llm": "CoreML/ANE"}
    color = {"litert-lm": "#e11d48", "mlx-swift": PALETTE["mlx-swift"],
             "llama.cpp": PALETTE["llama.cpp"], "coreml-llm": PALETTE["coreml-llm"]}

    dec: dict = {}
    mem: dict = {}
    for r in runs:
        lm = logical_model(r["model"]["id"])
        rt = r["runtime"]
        if lm not in models or rt not in runtimes:
            continue
        dec.setdefault((lm, rt), []).append(r["metrics"]["decodeTokensPerSecond"])
        mem.setdefault((lm, rt), []).append(r["metrics"]["memoryPeakDuringDecodeMB"])

    xs = list(range(len(models)))
    width = 0.20
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    def grouped(ax, data, fmt, title, ylab):
        ax.grid(False)
        labelled: set = set()
        for i, rt in enumerate(runtimes):
            for j, m in enumerate(models):
                vals = data.get((m, rt))
                if not vals:
                    continue  # e.g. no LiteRT Qwen row captured yet (Qwen3-0.6B coming)
                y = median(vals)
                x = xs[j] + (i - 1.5) * width
                ax.bar([x], [y], width,
                       label=(rt_label[rt] if rt not in labelled else None),
                       color=color[rt], edgecolor="white", linewidth=0.6)
                labelled.add(rt)
                ax.text(x, y, fmt.format(y), ha="center", va="bottom",
                        fontsize=10, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.margins(y=0.22)

    grouped(ax1, dec, "{:.0f}", "Decode throughput   ↑ better", "tok/s")
    grouped(ax2, mem, "{:.0f}", "Peak memory   ↓ better", "MB")
    ax1.legend(frameon=False, loc="upper right", fontsize=10, ncol=3,
               bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(
        "On-device LLM — iPhone 17 Pro (A19 Pro) · 4-bit · short-chat · median of 3 cold runs",
        fontsize=12.5, fontweight="bold", y=1.03,
    )
    fig.text(0.5, -0.03,
             "Decode: LiteRT-LM wins Gemma, MLX wins Qwen  ·  Memory: CoreML/ANE as low as 241 MB (Qwen, ~5× leaner)  ·  LiteRT-LM Qwen3 row coming; CoreML/ANE trades speed for memory",
             ha="center", fontsize=8.5, color="#666")
    plt.tight_layout()
    plt.savefig(OUT / "iphone_decode_mem.png")
    plt.close(fig)
    print(f"wrote {OUT / 'iphone_decode_mem.png'}")


# ------------------------------------------------------------------ #
#  Chart 5/6 — iPhone 17 Pro energy: J/token bar + speed×efficiency scatter
# ------------------------------------------------------------------ #

IPHONE_COLOR = {
    "litert-lm": "#e11d48", "mlx-swift": PALETTE["mlx-swift"],
    "llama.cpp": PALETTE["llama.cpp"], "coreml-llm": PALETTE["coreml-llm"],
}
IPHONE_RT_LABEL = {
    "litert-lm": "LiteRT-LM", "mlx-swift": "MLX-Swift",
    "llama.cpp": "llama.cpp", "coreml-llm": "CoreML/ANE",
}


def iphone_energy_points(model: str = "Gemma 4 E2B") -> dict[str, dict]:
    """Median energy stats per runtime from iphone17pro `energy`-task runs.

    Returns {} when no battery-delta energy rows exist yet, so the chart
    functions can skip cleanly until the measurements land."""
    runs = load_runs("iphone17pro", task="energy")
    agg: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        m = r.get("metrics") or {}
        if m.get("energySource") != "battery-1pct":
            continue
        if m.get("energyJoulesPerToken") is None:
            continue
        if logical_model((r.get("model") or {}).get("id", "")) != model:
            continue
        rt = r.get("runtime")
        d = agg.setdefault(rt, {"jpt": [], "tok_s": [], "w": [], "tok_wh": []})
        d["jpt"].append(m["energyJoulesPerToken"])
        d["tok_s"].append(m.get("decodeTokensPerSecond") or 0)
        d["w"].append(m.get("averagePackagePowerW") or 0)
        j, gt = m.get("energyJoules") or 0, m.get("generatedTokenCount") or 0
        if j > 0 and gt > 0:
            d["tok_wh"].append(gt * 3600.0 / j)
    out: dict[str, dict] = {}
    for rt, d in agg.items():
        out[rt] = {
            "jpt": median(d["jpt"]), "tok_s": median(d["tok_s"]),
            "w": median(d["w"]),
            "tok_wh": median(d["tok_wh"]) if d["tok_wh"] else None,
        }
    return out


def chart_iphone_energy():
    pts = iphone_energy_points()
    if not pts:
        print("skip iphone_energy_per_token (no battery-delta energy rows yet)")
        return
    order = ["litert-lm", "mlx-swift", "llama.cpp", "coreml-llm"]
    labels = [r for r in order if r in pts]
    jpt = [pts[r]["jpt"] for r in labels]

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    colors = [IPHONE_COLOR[r] for r in labels]
    bars = ax.barh(range(len(labels)), jpt, color=colors, edgecolor="white", linewidth=0.6)
    for bar, v, r in zip(bars, jpt, labels):
        tw = pts[r]["tok_wh"]
        tail = f"  ·  {pts[r]['w']:.1f} W avg" + (f"  ·  {tw:.0f} tok/Wh" if tw else "")
        ax.text(v + max(jpt) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f} J/tok{tail}", va="center", fontsize=9.5, color="#222")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([IPHONE_RT_LABEL[r] for r in labels], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Joules per generated token (lower = better)")
    ax.set_xlim(0, max(jpt) * 1.6)
    ax.set_title("Energy per token — iPhone 17 Pro, Gemma 4 E2B, sustained (battery-delta)")
    fig.text(0.5, -0.06,
             "iPhone battery-delta (1% steps), unplugged. Whole-system incl. display — compare runtimes, not against the Mac powermetrics rows.",
             ha="center", fontsize=8.5, color="#555")
    plt.savefig(OUT / "iphone_energy_per_token.png")
    plt.close(fig)
    print(f"wrote {OUT / 'iphone_energy_per_token.png'}")


def chart_iphone_tradeoff():
    pts = iphone_energy_points()
    if not pts:
        print("skip iphone_tradeoff (no battery-delta energy rows yet)")
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for rt, d in pts.items():
        ax.scatter(d["tok_s"], d["jpt"], s=240, color=IPHONE_COLOR.get(rt, "#888"),
                   edgecolor="white", linewidth=1.4, zorder=5)
        ax.annotate(IPHONE_RT_LABEL.get(rt, rt), (d["tok_s"], d["jpt"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=11, fontweight="bold", color="#222")
        ax.annotate(f"{d['jpt']:.2f} J/tok · {d['tok_s']:.0f} tok/s · {d['w']:.1f} W",
                    (d["tok_s"], d["jpt"]), textcoords="offset points",
                    xytext=(8, -14), fontsize=8.5, color="#555")

    ax.set_xlabel("Sustained decode (tok/s) — right = faster")
    ax.set_ylabel("Energy per token (J/tok) — down = more efficient")
    ax.set_title("Throughput × Energy — iPhone 17 Pro, Gemma 4 E2B (battery-delta)")
    ax.invert_yaxis()
    ax.set_xlim(0, max(d["tok_s"] for d in pts.values()) * 1.25)
    ax.set_ylim(max(d["jpt"] for d in pts.values()) * 1.2, 0)
    fig.text(0.5, -0.02,
             "Does the low-power ANE win J/token despite slow decode, or does GPU speed pay for itself? The phone's answer.",
             ha="center", fontsize=8.5, color="#555")
    plt.savefig(OUT / "iphone_tradeoff.png")
    plt.close(fig)
    print(f"wrote {OUT / 'iphone_tradeoff.png'}")


def main():
    chart_iphone()
    chart_iphone_energy()
    chart_iphone_tradeoff()
    chart_decode_tok_per_s()
    chart_energy_per_token()
    chart_itl_jitter()
    chart_tradeoff()
    print(f"\nAll charts in {OUT.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
