#!/usr/bin/env python3
"""Score the `quality` task outputs for correctness + degeneracy → docs/litert-lm/QUALITY.md.

NOT a perplexity / MMLU eval. A guardrail that catches quantization-induced quality *collapse*
(wrong answers, degenerate repetition) so the decode-tok/s comparison across runtimes — each with
its own native 4-bit quant — is read at roughly equal output quality, not speed-at-any-quality.
Coarse by design (substring match on the final answer, the text after any reasoning; 0.9 vs 0.11 is read
on its own line, see line_says_larger); a collapsed quant misses many, not one. A reply that ends inside
its reasoning (cut at the token limit) has no final answer and is shown as "no answer".

    python3 scripts/quality_check.py
"""
import json, glob, re, unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw"
OUT = REPO / "docs" / "litert-lm" / "QUALITY.md"

# (label, regex) — greedy answers to the 8 fixed questions in QualityTask.swift.
CHECKS = [
    ("17+25=42",        r"\b42\b"),
    ("capital=Tokyo",   r"tokyo"),
    ("opp(hot)=cold",   r"\bcold\b"),
    ("days/week=7",     r"\bseven\b|\b7\b"),
    ("thanks(fr)=merci", r"merci"),
    ("8*7=56",          r"\b56\b"),
    ("0.9>0.11",        r"0\.9"),
    ("rhyme=blue",      r"\bblue\b"),
]


# _number, _plain, larger_claims, says_larger, THOUGHT_TAGS and strip_think: copied from john-rocky/hf-to-litertlm scripts/verify_quality.py at commit ebc0684.
def _number(value):
    """A regex for one decimal as answers write it: 0.9 also as 0.90 and .9, never inside 10.9 or 0.95."""
    whole, frac = value.split(".")
    lead = "(?:0)?" if whole == "0" else whole
    return rf"(?<![\d.]){lead}\.{frac}0*(?!\d)"


def _plain(text):
    """Lowercase text without markdown and LaTeX marks, full-width forms folded (０．９ -> 0.9), a boxed
    result read as "answer: …", and without questions: a sentence that ends in "?" ("Which is larger: 0.9
    or 0.11?") claims nothing."""
    t = unicodedata.normalize("NFKC", text).lower()
    inner = r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}"
    while re.search(inner, t):
        t = re.sub(inner, r"\1", t)
    t = re.sub(r"\\boxed\{([^{}]*)\}", r" answer: \1 ", t)
    for a, b in (("\\geq", "≥"), ("\\ge", "≥"), ("\\gt", ">"), ("\\leq", "≤"), ("\\le", "≤"), ("\\lt", "<"),
                 ("&gt;", ">"), ("&lt;", "<")):
        t = t.replace(a, b)
    t = re.sub(r"[*`$\\]", " ", t)
    # sentences end at . ? ! 。 or a newline; a "." before a digit belongs to a number (0.9)
    sentences = re.findall(r"(?:[^.?!\n。]|\.(?=\d))*[.?!\n。]|(?:[^.?!\n。]|\.(?=\d))+$", t)
    return re.sub(r"[ \t]+", " ", "".join(s for s in sentences if not s.endswith("?")))


def larger_claims(answer, big, small):
    """[(position, number)] for every place the answer says which of the two numbers is larger, in the text
    says_larger() reads (lowercased, see _plain)."""
    t = _plain(answer)
    num = {big: _number(big), small: _number(small)}
    up = r"(?:larger|lager|bigger|greater|higher|more\s+than)"
    down = r"(?:smaller|lesser|less|lower)"
    # between a number and its verb: same sentence, no other number outside parentheses, no new clause
    # ("0.9 is larger than 0.11 because 9 is greater than 1" claims nothing about 0.11)
    gap = (r"(?:(?!\b(?:because|since|while|but|whereas|although|though)\b)[^.?!\n<>。\d(]"
           r"|\([^()\n]{0,30}\)){0,40}?")
    verb = r"(?:(?:is|was|are)(?:n't)?|'s|would be|will be|remains)"
    claims = []
    for x, y in ((big, small), (small, big)):
        X, Y = num[x], num[y]
        rules = (  # (regex, the number it says is larger, the number when a "not" turns it over)
            (rf"{X}{gap}\b{verb}\s+((?:\w+\s+){{0,2}}?){up}\b", x, y),
            (rf"{X}\s+{up}\s+than\b", x, None),
            (rf"{X}\s*\(\s*(?:the\s+)?{up}\s*\)", x, None),
            (rf"(?<!which is )(?<!which one is )\b{up}(?:\s+\w+){{0,2}}?\s*(?:is|:|=|would be)"
             rf"\s*(?:the\s+)?{X}(?!\s*(?:or|and|vs\.?|versus)\b)", x, None),
            (rf"\banswer\s*(?:is|:|=|would be)\s*(?:the\s+)?(?:number\s+)?{X}(?!\s*(?:or|and|vs\.?|versus)\b)", x, None),
            (rf"{X}\s*(?:>|≥|>=)\s*{Y}", x, None),
            (rf"{X}\s*(?:<|≤|<=)\s*{Y}", y, None),
            (rf"{X}{gap}\b{verb}\s+((?:\w+\s+){{0,2}}?){down}\b", y, x),
            (rf"{X}\s+{down}\s+than\b", y, None),
            (rf"{X}\s*\(\s*(?:the\s+)?{down}\s*\)", y, None),
            (rf"(?<!which is )\b(?:smaller|lesser)(?:\s+\w+){{0,2}}?\s*(?:is|:|=)\s*(?:the\s+)?{X}"
             rf"(?!\s*(?:or|and|vs\.?|versus)\b)", y, None),
            (rf"{X}\s*(?:の方|のほう|方)?\s*(?:が|は)\s*(?:より)?(?:大き|おおき)", x, None),
            (rf"{X}\s*(?:は|が)\s*{Y}\s*より(?:も)?\s*(?:大き|おおき)", x, None),
            (rf"{X}\s*(?:の方|のほう|方)?\s*(?:が|は)\s*(?:より)?(?:小さ|ちいさ)", y, None),
            (rf"{X}\s*(?:は|が)\s*{Y}\s*より(?:も)?\s*(?:小さ|ちいさ)", y, None),
            (rf"(?:大きい|大きな)(?:のは|方は|ほうは|数は|方が|ほうが|は|が|:)\s*{X}", x, None),
        )
        for rx, winner, if_not in rules:
            for m in re.finditer(rx, t):
                negated = if_not is not None and re.search(r"\bnot\b|n't", m.group(0))
                claims.append((m.start(), if_not if negated else winner))
    return sorted(claims)


def says_larger(answer, big, small):
    """True when the answer says `big` is the larger of the two numbers. The last claim decides:
    "A is larger / bigger / greater / higher / more than [B]", "A > B", "B < A", "the larger (one / number)
    is A", "the answer is A" (a boxed result too), "A (larger)", "B is smaller / less / lower", "B (smaller)",
    "the smaller is B", a "not" in such a claim turns it over, and the Japanese "A の方が大きい" /
    "A は B より大きい" / "B の方が小さい". An answer without a claim counts when it names only `big`
    ("0.9", "**0.9**"); naming both numbers without a claim is not an answer."""
    claims = larger_claims(answer, big, small)
    if claims:
        return claims[-1][1] == big
    t = _plain(answer)
    named = {v for v in (big, small) if re.search(_number(v), t)}
    return named == {big}


THOUGHT_TAGS = (("<think>", "</think>"), ("[thought]", "[/thought]"), ("[reasoning]", "[/reasoning]"))


def strip_think(text):
    """The final answer: the reply without its reasoning, so the check never reads words of the thought.
    Everything up to the last closing tag is reasoning (a template that opens <think> in the prompt leaves
    only </think> in the reply; the litert-lm CLI prints the thought channel as [thought] ... [/thought]).
    Returns "" when a thought opens and does not close (cut off mid-thought = no answer)."""
    for opening, closing in THOUGHT_TAGS:
        if closing in text:
            text = text.rsplit(closing, 1)[1]
        if opening in text:
            return ""
    return text.strip()


def line_says_larger(answer, big, small):
    """The answer to "Which is larger: big or small?" read on its own line, as the prompt asks for one answer
    per line: the first line that names either number outside a question decides, by its claim or by the one
    number it names (says_larger); a line that names both numbers without a claim is no answer."""
    for line in answer.splitlines():
        t = _plain(line)
        if re.search(_number(big), t) or re.search(_number(small), t):
            return says_larger(line, big, small)
    return False


def ends_in_reasoning(text):
    """True when the reply opens a reasoning block and ends before closing it (cut at the token limit), the
    case strip_think() returns "" for: such a reply has no final answer."""
    for opening, closing in THOUGHT_TAGS:
        if closing in text:
            text = text.rsplit(closing, 1)[1]
        if opening in text:
            return True
    return False


def correctness(text):
    """The 8 checks on the final answer (the text after any reasoning, strip_think). The 0.9 vs 0.11 regex
    also finds "0.9" in the wrong answer "0.11 is larger than 0.9.", so that check reads its own line."""
    answer = strip_think(text)
    t = answer.lower()
    return [line_says_larger(answer, "0.9", "0.11") if label == "0.9>0.11" else bool(re.search(p, t))
            for label, p in CHECKS]


def degenerate(text):
    """True if the output loops or is special-token spam."""
    words = text.split()
    if len(words) >= 10:
        grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
        if grams and Counter(grams).most_common(1)[0][1] >= 3:
            return True
        if len(set(words)) / len(words) < 0.30:
            return True
    # No-space loops (e.g. "<|fim_prefix|><|fim_prefix|>…"): tiny char diversity, or a
    # special token leaked + repeated. split()-by-space sees these as one "word", so check chars.
    if len(text) >= 40 and len(set(text)) < 15:
        return True
    if text.count("<|") >= 5 or text.count("<pad>") >= 5:
        return True
    return False


def empty_collapse(text, gen_tokens, streamed_chunks):
    """A run that reported decode tokens but streamed (almost) no text — the
    model emitted only special / non-decodable tokens. The substring checks
    can't see this (there is nothing to match), so it would otherwise score
    0/8 + "not degenerate", masking a collapse as a mere wrong-answer run.
    `streamed_chunks` is None on pre-2026-06 JSONL (field absent) — fall back
    to the output length in that case."""
    if streamed_chunks == 0 and gen_tokens and gen_tokens > 0:
        return True
    if gen_tokens and gen_tokens >= 8 and len(text.strip()) < 5:
        return True
    return False


def main():
    rows = []
    for f in sorted(RAW.glob("*-quality-run*.jsonl")):
        raw = open(f).read()
        try:
            d = json.loads(raw.split("\n")[0])  # JSONL: first record
        except json.JSONDecodeError:
            d = json.loads(raw)                 # pretty-printed single record (import_to_flat.py)
        dev = f.name.split("-quality-")[0]
        out = d.get("outputSample", "")
        m = d.get("metrics", {})
        gen = m.get("generatedTokenCount", 0)
        streamed = m.get("streamedChunkCount")  # None on pre-2026-06 rows
        hits = correctness(out)
        # a reply cut inside its reasoning has no final answer: say so instead of a count
        limit = gen or d.get("parameters", {}).get("maxTokens")
        no_answer = ("no answer: the reply ends inside the reasoning" + (f" at {limit} tokens" if limit else "")
                     if ends_in_reasoning(out) else None)
        rows.append({
            "dev": dev,
            "runtime": d.get("runtime", "?"),
            "model": d.get("model", {}).get("displayName", d.get("model", {}).get("id", "?")),
            "quant": d.get("model", {}).get("quantization", "?"),
            "score": sum(hits), "hits": hits, "no_answer": no_answer,
            "degen": degenerate(out) or empty_collapse(out, gen, streamed),
            "sample": " ".join(out.split())[:300] or f"(no text — {gen} tokens, 0 decoded)",
        })

    lines = [
        "# Quality parity — correctness + degeneracy guardrail",
        "",
        "> Scored by [`scripts/quality_check.py`](../../scripts/quality_check.py) from the `quality` task "
        "(8 fixed checkable questions, greedy). **Not** a perplexity/MMLU eval — a guardrail that catches "
        "quantization-induced quality *collapse* so the decode-tok/s tables compare runtimes at roughly "
        "equal quality, not speed-at-any-quality. Each runtime uses its native 4-bit quant (disclosed).",
        "",
        "| Device | Runtime | Model | Quant | Correct | Degenerate? |",
        "|---|---|---|---|---:|:--:|",
    ]
    for r in rows:
        flag = "⚠️ yes" if r["degen"] else "no"
        correct = r["no_answer"] or f"{r['score']}/8"
        lines.append(f"| {r['dev']} | {r['runtime']} | {r['model']} | {r['quant']} | "
                     f"{correct} | {flag} |")
    if not rows:
        lines.append("| _(no quality runs yet — run `--task quality`)_ | | | | | |")
    lines += ["", "## Per-question hits", "",
              "`" + "  ".join(c[0] for c in CHECKS) + "`", ""]
    for r in rows:
        marks = "no answer" if r["no_answer"] else "`" + "".join("✓" if h else "·" for h in r["hits"]) + "`"
        lines.append(f"- **{r['runtime']} {r['model']}** ({r['dev']}): {marks}  — “{r['sample']}”")
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT.relative_to(REPO), f"({len(rows)} runs)")


if __name__ == "__main__":
    main()
