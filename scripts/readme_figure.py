#!/usr/bin/env python3
"""Draw the figure at the top of README.md from the Gemma 4 E2B table in docs/results-iphone.md.

    python3 scripts/readme_figure.py

Writes docs/images/gemma4-e2b-iphone-light.svg and -dark.svg, and prints the sentence the figure's alt text uses.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "docs", "results-iphone.md")
OUT = os.path.join(ROOT, "docs", "images")
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": {"text": "#0b0b0b", "muted": "#52514e", "bar": "#2a78d6", "other": "#eb6834"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "bar": "#3987e5", "other": "#d95926"},
}
HEADER = "| Runtime | Build | Decode tok/s |"
# (start of the Build column, label in the figure)
BUILDS = [
    ("wNa8o8 QAT", "LiteRT-LM, wNa8o8 QAT"),
    ("CQ4 **uncalibrated**", "Cactus, CQ4 uncalibrated"),
    ("**CQ4 as shipped**", "Cactus, CQ4 as shipped"),
    ("PTQ 4-bit", "MLX Swift, PTQ 4-bit"),
    ("own int4", "Core AI, own int4, patched engine"),
    ("Q4_K_M", "llama.cpp, Q4_K_M"),
    ("QAT OptiQ", "MLX Swift, QAT OptiQ int4"),
]
TITLE = "Gemma 4 E2B on an iPhone 17 Pro, one row per build"
NOTES = [
    "Rows are grouped by capture session, July 2026 on iOS 27. Compare decode and memory only within one group.",
    "GSM8K: n=100, measured on an M4 Max with one harness. LiteRT-LM had a 2,048-token limit, the others 1,024.",
    "Higher decode and GSM8K are better, lower memory is better.",
]
LEGEND = [("bar", "Warm median, memory charged to the app"), ("other", "Other basis, as labeled: cold run, peak memory, memory-mapped")]
WIDTH, DATES, LABELS, PANEL_GAP, VALUE_ROOM, ROW, BAR, GROUP_GAP = 860, 44, 284, 14, 84, 34, 18, 14


def number(cell, what):
    match = re.search(r"\d[\d,]*(?:\.\d+)?", cell.replace("*", ""))
    if match is None:
        sys.exit(f"readme_figure: no {what} in the cell {cell!r}")
    return float(match.group(0).replace(",", ""))


def rows():
    """Return [{label, session, decode, memory, gsm8k, decode_other, memory_other}] read from the table, by session."""
    if not os.path.exists(SOURCE):
        sys.exit(f"readme_figure: {SOURCE} not found")
    with open(SOURCE, encoding="utf-8") as f:
        lines = f.read().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(HEADER)), None)
    if start is None:
        sys.exit(f"readme_figure: no table starting {HEADER!r} in docs/results-iphone.md")
    table = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        table.append([c.strip() for c in line.strip("|").split("|")])
    out = []
    for key, label in BUILDS:
        cells = next((c for c in table if c[1].startswith(key)), None)
        if cells is None:
            sys.exit(f"readme_figure: no row for the build {key!r} in docs/results-iphone.md")
        session = re.search(r"\b\d{1,2}/\d{1,2}\b", cells[2])
        if session is None:
            sys.exit(f"readme_figure: no capture session in the cell {cells[2]!r}")
        out.append({
            "label": label,
            "session": session.group(0),
            "decode": number(cells[2], "decode number"),
            "memory": number(cells[4], "memory number"),
            "gsm8k": number(cells[5], "GSM8K score"),
            "decode_other": "cold" if "cold" in cells[2] else "",
            "memory_other": "peak" if "peak basis" in cells[4] else "mmap" if "†" in cells[4] else "",
        })
    out.sort(key=lambda r: ([int(p) for p in r["session"].split("/")], -r["decode"]))
    return out


def text(x, y, body, size, fill, weight="400"):
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" fill="{fill}">{body}</text>'


def tops(data, first):
    """Return the y of each row, with a gap between capture sessions."""
    out, y = [], first
    for n, r in enumerate(data):
        if n and r["session"] != data[n - 1]["session"]:
            y += GROUP_GAP
        out.append(y)
        y += ROW
    return out


def draw(data, theme, desc):
    c = THEMES[theme]
    parts = [text(0, 28, TITLE, 20, c["text"], "600")]
    y = 52
    for note in NOTES:
        parts.append(text(0, y, note, 14, c["muted"]))
        y += 20
    y += 16
    panels = [("Decode, tok/s", "decode", "decode_other", "{:.1f}"), ("Memory, MB", "memory", "memory_other", "{:,.0f}"), ("GSM8K, %", "gsm8k", None, "{:g}")]
    width = (WIDTH - LABELS - PANEL_GAP * (len(panels) - 1)) / len(panels)
    row_tops = tops(data, y + 14)
    for i, (title, key, other, shape) in enumerate(panels):
        x = LABELS + i * (width + PANEL_GAP)
        parts.append(text(x, y, title, 15, c["text"], "600"))
        unit = (width - VALUE_ROOM) / max(r[key] for r in data)
        for top, r in zip(row_tops, data):
            fill = c["other"] if other and r[other] else c["bar"]
            w = max(r[key] * unit, 2)
            parts.append(f'<rect x="{x:g}" y="{top:g}" width="{w:g}" height="{BAR}" rx="4" fill="{fill}"/>')
            suffix = f" {r[other]}" if other and r[other] else ""
            parts.append(text(x + w + 6, top + 14, shape.format(r[key]) + suffix, 14, c["text"], "600"))
    for n, (top, r) in enumerate(zip(row_tops, data)):
        if n == 0 or r["session"] != data[n - 1]["session"]:
            parts.append(text(0, top + 14, r["session"], 14, c["muted"], "600"))
        parts.append(text(DATES, top + 14, r["label"], 14, c["text"]))
    y = row_tops[-1] + ROW + 12
    x = 0
    for slot, label in LEGEND:
        parts.append(f'<rect x="{x}" y="{y - 11}" width="12" height="12" rx="3" fill="{c[slot]}"/>')
        parts.append(text(x + 18, y, label, 14, c["muted"]))
        x += 18 + 7 * len(label) + 28
    height = y + 10
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:g}" width="{WIDTH}" height="{height:g}" '
        f'role="img" aria-labelledby="t d" font-family="{FONT}">\n'
        f'<title id="t">{TITLE}</title>\n<desc id="d">{desc}</desc>\n' + "\n".join(parts) + "\n</svg>\n"
    )


def main():
    data = rows()
    def basis(word):
        return f" {word}" if word else ""

    desc = TITLE + ", grouped by capture session. " + " ".join(
        f"Session {r['session']}, {r['label']}: {r['decode']:.1f} tok/s{basis(r['decode_other'])}, "
        f"{r['memory']:,.0f} MB{basis(r['memory_other'])}, GSM8K {r['gsm8k']:g}%." for r in data)
    os.makedirs(OUT, exist_ok=True)
    for theme in THEMES:
        with open(os.path.join(OUT, f"gemma4-e2b-iphone-{theme}.svg"), "w", encoding="utf-8") as f:
            f.write(draw(data, theme, desc))
    print(desc)


if __name__ == "__main__":
    main()
