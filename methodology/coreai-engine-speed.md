# Core AI engine speed — the pipelined-GPU finding (and the zoo paradox)

## The finding (M4 Max, Qwen3-0.6B, 4-bit, decode)

| Engine | Decode tok/s | Path |
| --- | ---: | --- |
| **Core AI** (`coreai-pipelined`, GPU) | **~1,150** | official `coreai.llm.export qwen3-0.6b` → 327 MB INT4 dynamic `.aimodel` → `CoreAILM` `EngineFactory` (auto-selects the pipelined GPU engine) |
| **MLX** (Metal, GPU) | **~535** | `mlx-community/Qwen3-0.6B-4bit` via `mlx_lm` |

**Core AI's pipelined GPU engine decodes ~2.1× faster than MLX** on this model.
Credible: 327 MB ≈ 4-bit, and ~1,150 tok/s is under the M4 Max 4-bit
memory-bandwidth ceiling (~1,800 tok/s). Core AI even ran the *longer* context
(128-token synthetic prompt vs MLX's 23), so the gap is if anything understated.

### Reproduce
```bash
# Core AI (official microbench tool)
cd ~/code/coreai/coreai-models
DEVELOPER_DIR=/path/to/Xcode-beta.app/Contents/Developer \
  xcrun swift run -c release llm-benchmark \
    --model exports/qwen3_0_6b_dynamic -p 128 -g 256 -n 3

# MLX
python -m mlx_lm generate --model mlx-community/Qwen3-0.6B-4bit \
    --prompt "<~128 tokens>" --max-tokens 256 --temp 0.0
```

**Caveat:** different tools / prompts (synthetic vs real). The ~2× signal is
robust; an exact figure needs both run through one harness (yardstick) with the
same prompt. TODO.

## The paradox to investigate

In **coreai-model-zoo**, the community **Gemma 4 E2B / Qwen 3.5** Core AI models
benchmark **~3× SLOWER than MLX** — the *opposite* of the result above.

**Same framework (Core AI), opposite outcome.** Working hypothesis: the official
Qwen3-0.6B bundle runs through the optimized **`coreai-pipelined`** engine
(double-buffered GPU dispatch), while the zoo models run through **hand-rolled
per-token backends** (CoreAIShared `PreparedModel` + synchronous `predict()` per
step, no double-buffering) — and/or their head-split / host-cache / SSM
architectures force a slower path that the pipelined engine can't drive.

If the bottleneck is the *execution path* (not the architecture), re-routing the
zoo models through the pipelined engine should bring the same ~2× advantage.

## Root cause (confirmed)

**The zoo models bypass Apple's optimized runtime.** The official Qwen3-0.6B
rides `CoreAILanguageModel` → `EngineFactory` → the **`coreai-pipelined` GPU
engine**. The zoo Gemma4/Qwen3.5 ports hand-roll a **synchronous per-token
dispatch loop** over the low-level `AIModel.load()` / `fn.run()` primitives
(`CoreAIChat/Gemma4MonolithEngine.swift`, `QwenChatFast/FastEngine.swift`).

What the pipelined engine does that the hand-rolled loop doesn't:

1. **Non-blocking async encode + callback sampling** (`CoreAIPipelinedEngine.swift`).
   CPU submits GPU work via `function.encode(...)` and immediately preps the next
   token; a `PipelineGate(capacity: 3)` bounds in-flight steps; the sampler's
   completion handler (Metal GPU-completion thread) yields the token. GPU step N
   overlaps CPU prep for N+1. The zoo loop `await fn.run()` **blocks on GPU
   completion every token** — zero overlap.
2. **On-GPU sampling** (`MPSGraphSamplers.swift` argmax/topK). The 600 KB-vocab
   (Gemma4: **262 K-vocab**) logits never leave the GPU — argmax runs on-GPU, the
   token lands in a GPU buffer. The zoo loop reads logits back to CPU and samples
   on CPU (the 262 K-logit readback is Gemma4's single biggest per-token cost,
   ~125 ms → 69 ms even with an in-graph argmax kernel).
3. **On-device stateful KV + double-buffering.** Pipelined grows the KV by
   buffer-expand + async blit (no indexed writes). The zoo uses **host-cache**:
   the CPU re-feeds the whole KV every token — because the in-graph KV write
   (`mutable_slice_update`) **SIGSEGVs on the Core AI WWDC26 beta**. Plus Gemma4
   = 2 dispatches/token (core + head), Qwen3.5-chunked = 4 (per chunk).

Also: the "static int8" exports were meant for the ANE but the **beta ANE
compiler rejects them** (`MLIR MPS→ANEC failed`) → silent **GPU fallback** —
so they get neither ANE efficiency nor the pipelined-GPU fast path.

### Measured (iPhone 17 Pro, from `ondevice/*_RESULTS.md`)
| Model · path | tok/s | vs MLX/CoreML |
| --- | ---: | --- |
| Gemma4 E2B — GPU monolith int8, hand-rolled | 22–24 | ~1.5× slower (CoreML 34) |
| Qwen3.5-0.8B — GPU monolith int8, hand-rolled | ~44 | ~1.1× slower (CoreML 48) |
| Qwen3.5-0.8B — ANE dynamic (shipped) | **14.7** | **~3.3× slower** |
| Qwen3-0.6B — official pipelined (this bench, Mac GPU) | — | **~2× faster** |

So even the zoo's *best* hand-rolled path is ~parity; the shipped ANE path is the
3× loss. The pipelined engine is the difference between ~parity and ~2× faster.

## Fix paths (to bring the advantage to the zoo)

- **A — ride the official pipelined engine.** Export as ONE dynamic `main`
  bundle (embed in-graph, dynamic KV) and load via `CoreAILanguageModel` instead
  of the hand-rolled loop. Feasible for **Qwen3.5-0.8B** (254 MB embed fits
  in-graph; head-split was a size optimization, not a requirement) **iff** the
  pipelined engine can carry the SSM conv/rec states alongside KV. **Not**
  feasible for Gemma4 as-is (9.4 GB per-layer PLE table can't live in the graph).
- **B — adopt the three techniques in the custom backend.** Biggest quick win:
  **on-GPU MPSGraph sampling** (kills Gemma4's 262 K-logit readback). Then async
  non-blocking dispatch + callback (CPU/GPU overlap), and pipelined KV growth
  (buffer-expand+blit) to drop host-cache without hitting the in-graph-write
  SIGSEGV.

**Decisive next experiment:** export Qwen3.5-0.8B as a full (non-head-split)
dynamic bundle and run it through `llm-benchmark` (which uses the pipelined
engine). Fast → path A is proven; SSM error → the engine can't carry SSM and
path B is the route.
