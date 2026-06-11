# Core AI export-lowering sensitivity — same recipe, 2.2× different artifact

**TL;DR: `coreai.llm.export qwen3-0.6b` produced a 1,116 tok/s artifact when this
repo's Mac numbers were first taken, and a ~500 tok/s artifact two days later — same
command, same registry preset, same source checkout, same wheel versions, same
machine. The only environment change in between was the macOS 26 → 27 beta upgrade.
Benchmark the artifact you ship, pin the artifact, and don't assume a re-export
reproduces it.**

## The discrepancy

Preparing the official-recipe matrix (2026-06-11) we re-exported Qwen3-0.6B and
measured 484 tok/s (512p/1024g) where this repo had published ~1,121 (=1,150-class)
for the same recipe. A/B on the same day, same `llm-benchmark` release binary,
`-p 128 -g 256 -n 3`:

| Artifact | Exported | Decode tok/s | Prefill tok/s |
|---|---|---:|---:|
| `qwen3_0_6b_dynamic` (original) | 2026-06-09 (macOS 26) | **1,116** | **17,350** |
| `qwen3_0_6b_4bit_dynamic` (re-export) | 2026-06-11 (macOS 27 beta) | 500 | 6,667 |
| re-export, pristine upstream `main` @0c1055f | 2026-06-11 (macOS 27 beta) | 504 | 6,676 |

## What's actually different — the program, not the runtime

Both artifacts run on the *same* runtime in these measurements (macOS 27 beta,
same `llm-benchmark` binary) — so the runtime didn't regress; the exported
**program** differs. Op-level evidence (`strings main.mlirb`):

- **Fast artifact**: plain `Linear$N` composites, **zero** quantization ops in the
  program text, yet 327 MB (4-bit-sized) → 4-bit weights consumed natively by the
  runtime's Linear kernels (quantized-matmul path).
- **Slow artifact**: `ParametrizedLinear$N` composites + 141×
  `constexpr_blockwise_shift_scale` ops → explicit dequantize-then-matmul.

Same 4-bit storage class (327 vs 320 MB); the **compute path** differs 2.2×.

## What we ruled out (all held constant)

- Source code: local checkout `b1cb71b` unchanged (reflog: single clone, no resets);
  upstream `main` (0c1055f) differs only in `model_registry.py` (cosmetic). A pristine
  re-clone reproduces the slow artifact.
- Wheels: `coreai-core 1.0.0b1 / coreai-torch 0.4.0 / torch 2.9.0`, all installed
  2026-06-09 05:45, untouched since (file mtimes).
- Command: re-ran the original line verbatim (`uv run coreai.llm.export qwen3-0.6b
  --platform macOS --output-name …`; `uv run` resolves to the same project venv).
- Registry preset: `("qwen3-0.6b", …, "4bit", "float16", 8192)` in both revisions.

## The mechanism (deeper forensics, 2026-06-11 night)

Following the "how can the OS change a python export?" question, we traced the
pipeline component by component:

1. **The `coreai-core` wheel ships TWO complete native stacks** and picks one at
   import time (`coreai/runtime/__init__.py`): macOS < 27 → the wheel-bundled local
   stack (`_coreai_runtime.so`); macOS ≥ 27 + wheel install → the **OS framework**
   (`_coreai_runtime_os.so`). Env overrides exist: `USE_LOCAL_COREAI` /
   `USE_OS_COREAI`. The compiler bindings (`_coreaiIR`) ride the same switch.
   So the 06-09 export (macOS 26) compiled with the bundled stack; the 06-11
   exports compile with the OS framework by default.
2. **The quantizer is constant.** `quantize_pytorch_model` → `coreai-opt` PT2E
   `Quantizer` (ExecutionMode comes from static config, no environment probing) —
   it ALWAYS emits the parametrized/dequant form. The fast artifact's
   plain-`Linear$N`-no-dequant form must therefore be produced LATER, by the
   compiler folding dequant into the Linear composites during
   `prog.optimize()` (`coreai-pre-compilation-rewrite`) / serialization.
3. **The negative result that pins it on the OS:** re-exporting on macOS 27β with
   `USE_LOCAL_COREAI=1` — i.e. the *byte-identical frozen wheel compiler that
   produced the fast artifact on macOS 26* — STILL yields the dequant-style
   artifact (141 `constexpr_blockwise_shift_scale`, ~500 tok/s). Same pass code,
   different OS underneath, different lowering ⇒ the fold decision consults the
   running OS (capability/target queries under the pass), not just the stack's
   own code.

Everything userland is hash/mtime-verified constant (models repo, coreai-core /
coreai-torch / coreai-opt / torchao / transformers / torch wheels, command,
registry). The one moving part is macOS itself. We cannot bisect further without
a macOS-26 machine or Apple's pass sources; plausibly a 27-beta regression in
quantized-Linear legalization. Worth an Apple feedback with this document attached.

## Consequences for benchmarking (and for shipping)

1. **An `.aimodel` is a build artifact, not a pure function of the recipe.** Treat it
   like a compiled binary: version-stamp it, keep it, benchmark exactly what ships.
2. Numbers in this repo now carry the artifact date + OS in
   [`results/raw/2026-06-11-m4max-coreai-matrix/ENV.md`](../results/raw/2026-06-11-m4max-coreai-matrix/ENV.md).
3. The effect is size-dependent: at 8B both artifact generations measure ~94 tok/s
   (bandwidth-bound); at 0.6B the lowering dominates (2.2×). Small-model numbers are
   the canary.
4. If you have a macOS-26-era artifact, **keep it** — as of the 27 beta we know no
   recipe flag that re-produces the native-quantized lowering.
