# Long-context & continuous generation (decode-flatness)

The "does it hold up under load" axis: how prefill, decode rate, and memory behave as the prompt
(KV depth) grows, and over a long continuous decode. M4 Max, `yardstick` CLI, median/n=1, prompt
token counts are the runtime-recorded `promptTokenCount` (nominal `-8k`/`-32k` ids are approximate).
Data: `results/raw/m4max-*-long-context*.jsonl`, `*-sustained-generation-*.jsonl`.

## MLX (Metal GPU) — decode-flatness, decode tok/s @ prompt tokens (peak RAM)
| Model | continuous¹ | ~2.7K | ~10.7K | ~43K |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 474 (0.8 GB) | 390 (2.4 GB) | 219 (15.6 GB) | **84 (105 GB)** |
| Qwen3-4B | 151 (2.7 GB) | 138 (4.8 GB) | 101 (21.7 GB) | — (OOM) |
| Qwen3-8B | 93 (5.0 GB) | 88 (6.9 GB) | 72 (23.8 GB) | — (OOM) |

¹ `sustained-generation` = 512-token continuous decode from a ~20-token prompt (≈ zero-context
decode rate). Prefill tok/s also degrades with length (0.6B: 7050 → 5751 → 1934).

**Reading it:** decode falls off steeply with context (0.6B: 390 → 84, ~22% retained by 43K) and
**KV/attention memory explodes** — 105 GB for a *0.6B* model at 43K tokens, and ≥4B OOMs a 128 GB
machine past ~16K. Long context is memory-bound, not compute-bound, well before it's slow. (32K is a
0.6B-only ceiling probe for this reason; it's out of the routine sweep.)

## LiteRT-LM — two long-context blockers on macOS (findings)
1. **KV is pre-allocated to a fixed `maxNumTokens`, and any longer prompt is rejected** —
   `INVALID_ARGUMENT: Input token ids are too long: 2682 >= 512`. Unlike a dynamic-KV runtime, you
   must declare the max context up front. (Fixed in our adapter: `MediaPipeRuntime` now sizes
   `maxNumTokens` to prompt+output via `prepareContext`, not just the output budget — see the commit.)
2. **With the KV sized to fit the long prompt, generation then segfaults** (SIGSEGV / rc=139) on the
   macOS CPU path, at every size (0.6/4/8B), for both ~2.7K and ~10.7K prompts. So **litert
   long-context is currently unavailable on macOS** — it either rejects the prompt or crashes.

LiteRT **continuous** generation is fine, though: `sustained-generation` (512-token decode, short
prompt) runs at CPU decode 266 / 110 / 67 tok/s (0.6/4/8B) — i.e. decode is stable over a long
*output*; only a long *input* breaks it.

> The litert long-context blockers are macOS-CPU-path findings (same backend caveat as
> [`MACOS_DESKTOP.md`](MACOS_DESKTOP.md)); the iPhone Metal-GPU path may differ and is worth testing
> after the next on-device build. Adapter `prepareContext` is now correct for both platforms, so the
> iPhone run will tell us whether the cap→segfault is macOS-specific.

## For LiteRT
- Long context on the macOS build is a hard gap (reject-or-crash); decode-over-long-*output* is fine.
- The KV-pre-allocation model (declare `maxNumTokens` up front, reject longer) is a real design
  constraint for variable-context apps vs dynamic-KV runtimes — worth documenting either way.
