# LiteRT-LM v0.13.1 — sustained Qwen3 generation hangs / `DEADLINE_EXCEEDED` on iOS GPU

> Bug report for the `google-ai-edge/LiteRT-LM` team. Reverse-engineered from the **v0.13.1**
> source as vendored in this repo (`ios/BenchmarkApp/Vendored/LiteRT-LM`, `version.bzl`
> `VERSION = "0.13.1"`) plus the observed on-device symptom. Causes below are labelled as
> **hypotheses** — we read the code to ground them, but have not isolated a single root cause
> from a full stack trace. Part of the neutral [Apple Silicon LLM Benchmark](../../README.md);
> kept as a documented failure under [fairness rule 4](../../methodology/fairness-rules.md)
> ("failed runs stay in the table"). The same adapter's short-chat run passes, so this is a
> LiteRT-LM finding, not a harness bug.

## TL;DR

On iPhone 17 Pro / iOS 27, the Metal **GPU** backend of LiteRT-LM v0.13.1 completes a 128-token
short-chat turn for `litert-community/Qwen3-0.6B` normally (~119 tok/s), but a **600 s continuous
generation** (our energy / throttle task) **does not complete**: generation stalls and the turn
fails with `DEADLINE_EXCEEDED` whose message names the **`callback_thread_pool`**. Two console
signals accompany it:

1. `libLiteRtTopKMetalSampler.dylib` does **not** load (dlopen), so sampling silently falls back
   to the **CPU** top-k sampler.
2. The failure surfaces only on the long, continuous turn — never on the short 128-token turn with
   an otherwise identical configuration.

The code shows **two independent deadline surfaces** that a 600 s turn collides with, plus a
**CPU-sampling fallback** that gets more expensive the longer the turn runs. We believe the hang is
the interaction of these, not a single bug.

## Environment

| | |
|---|---|
| **Runtime** | `google-ai-edge/LiteRT-LM` **v0.13.1** (SPM product `LiteRTLM`; vendored tag `v0.13.1`, `version.bzl` `VERSION = "0.13.1"`) |
| **Backend** | Metal **GPU** (`EngineConfig(backend: .gpu)`) |
| **Sampler** | `SamplerConfig(topK: 40, topP: 1.0, temperature: 0)` — a real top-k sampler, so the GPU top-k sampler path is exercised (not the executor-internal shortcut) |
| **Model** | `litert-community/Qwen3-0.6B` → `qwen3_0_6b_mixed_int4.litertlm` (mixed INT4, blockwise gs32, 498 MB on disk) |
| **Device** | iPhone 17 Pro (`iPhone18,1`, A19 Pro, 8 GB), **iOS 27.0**, **Release** build, unplugged |
| **Adapter** | [`MediaPipeRuntime.swift`](../../ios/BenchmarkApp/Sources/Runtimes/MediaPipeRuntime.swift) (kind `litert-lm`), driving `Engine` → `Conversation.sendMessageStream` |
| **Task** | `energy` — continuous generation for `--sustain-seconds 600` (≈ 70k decoded tokens at ~119 tok/s) |

## Symptom

- **Short-chat (128-token cap):** completes, ~119 tok/s decode, ~239 ms TTFT. Reproducible n=3.
- **Sustained (600 s continuous):** does not complete. The turn ends in
  `absl::DeadlineExceededError` whose text names the pool `callback_thread_pool`. No tokens are
  returned for the run; the energy/throttle row cannot be measured.
- A `dlopen`-failed warning for `libLiteRtTopKMetalSampler.dylib` is logged at session start in
  **both** cases, but only the sustained turn fails.

## Minimal reproduction

1. Add the LiteRT-LM v0.13.1 Swift product to an iOS app target; build **Release**; run on an
   iPhone 17 Pro / iOS 27 (any A-series GPU device should do).
2. `EngineConfig(modelPath: <qwen3_0_6b_mixed_int4.litertlm>, backend: .gpu, maxNumTokens: 1024)`,
   `engine.initialize()`.
3. `createConversation(with: ConversationConfig(samplerConfig: SamplerConfig(topK: 40, topP: 1.0, temperature: 0)))`.
4. **Short turn (passes):** `sendMessageStream(Message(prompt))`, break after 128 chunks → completes.
5. **Sustained turn (hangs/fails):** keep consuming `sendMessageStream` (re-prompting on EOS) for
   ≥ 600 s, i.e. our energy protocol
   ([`methodology/energy-ios.md`](../../methodology/energy-ios.md)):
   ```
   xcrun devicectl device process launch --device <udid> \
     com.iosllmbenchmark.benchmarkapp -- \
     --yardstick-autorun --runtime litert-lm \
     --model-id "litert-community/Qwen3-0.6B" \
     --task energy --sustain-seconds 600 --runs 1
   ```
   → stalls, then `DEADLINE_EXCEEDED` naming `callback_thread_pool`.

## Evidence from the v0.13.1 source

All paths are relative to the repo root; line numbers are the vendored v0.13.1 tree.

### A. A single-threaded callback pool drained by a hard 10-second deadline at task finish

`runtime/framework/resource_management/threaded_execution_manager.cc:74-79` — **both** the execution
pool and the callback pool are created with **one** worker thread:

```cpp
execution_thread_pool_ = std::make_unique<ThreadPool>("execution_thread_pool", /*max_num_threads=*/1);
callback_thread_pool_  = std::make_unique<ThreadPool>("callback_thread_pool",  /*max_num_threads=*/1);
```

`:434-447` — every finished task schedules its user callback onto that single-threaded pool; the
callback re-enters user code and then takes `session_and_task_lookup_mutex_` to `UpdateTaskState`.

`:455-459` — `FinishTask` then **blocks** on the callback pool with a fixed 10 s deadline, with an
explicit upstream TODO acknowledging the synchronous wait is a problem:

```cpp
if (callback_thread_pool_ != nullptr) {
  // TODO b/476205457 - Consider to use a asynchronous approach to handle the
  // callback, and remove this WaitUntilDone.
  RETURN_IF_ERROR(callback_thread_pool_->WaitUntilDone(absl::Seconds(10)));
}
```

`runtime/framework/threadpool.cc:138-153` — `WaitUntilDone` returns exactly the observed error,
**with the pool name embedded**:

```cpp
return absl::DeadlineExceededError(absl::StrCat(
    "Timeout waiting for all tasks to be done in pool '", name_prefix_,
    "'. Tasks still in queue: ", tasks_.size(),
    ", Active tasks: ", num_active_tasks_));
```

With `name_prefix_ == "callback_thread_pool"`, this is precisely the message seen on device.

### B. A whole-task ceiling that equals the run length

`runtime/engine/engine.h:335`:

```cpp
static constexpr absl::Duration kDefaultTimeout = absl::Minutes(10);   // == 600 s
```

`runtime/core/session_advanced.cc:81,215,300` — the synchronous `RunPrefill` / `RunDecode` block on
`task_controller->WaitUntilDone(Engine::kDefaultTimeout)`. So a single decode turn has a **hard
10-minute ceiling that is not configurable through the Swift API**. Our sustained task is **600 s by
design** — it sits exactly on this boundary. A turn that legitimately decodes for ≥ 10 minutes would
fail here regardless of throughput or the sampler issue. This alone blocks any single sustained turn
≥ 10 min.

### C. The Metal top-k sampler dlopen fails → silent CPU fallback

`runtime/components/sampler_factory.cc:512-519` — the Metal sampler is loaded by **name** via
`dlopen`, lazily, into a local namespace:

```cpp
auto capi_or = GetSamplerCApi("libLiteRtTopKMetalSampler.dylib", ...);   // SharedLibrary::Load(..., RtldFlags::Lazy().Local())
```

`:657-703` (`CreateGpuSampler`, non-Android Apple path) — if the Metal load returns `kUnavailable`
and no WebGPU/OpenCL sampler is compiled in (the iOS case), the function returns
`absl::UnavailableError("GPU sampler not available.")`.

`:707-741` (`CreateSampler`) — on a GPU sampler `kUnavailable`, it logs *"GPU sampler unavailable.
Falling back to CPU sampling…"* and falls through to `CreateCpuSampler` → `TopPSampler` (a **CPU**
top-p/top-k sampler). `runtime/components/sampler_factory_failed_dlopen_test.cc` asserts exactly this
GPU→CPU fallback. So when the dylib does not load, **every** sampled token is produced on the CPU,
which requires the per-token logits to leave the GPU. Over Qwen3's ~151k-token vocabulary this is a
non-trivial per-token cost.

> Why the dylib likely fails to load on iOS: `prebuilt/ios_arm64/libLiteRtTopKMetalSampler.dylib`
> ships in the package, but `dlopen("libLiteRtTopKMetalSampler.dylib")` by bare name only succeeds
> if that `.dylib` is embedded in the app bundle's `Frameworks/` and code-signed for the run
> destination. The SPM product does not appear to embed/sign it, so the lazy load fails and the CPU
> fallback engages. This is a **packaging/integration gap**, distinct from the deadlines above.

## Hypothesised cause (interaction, not a single bug)

Ordered by how directly the code supports them:

1. **The single-threaded `callback_thread_pool` cannot drain within its 10 s deadline under
   sustained load (A).** A 600 s Qwen3 turn streams ~70k chunks; each schedules a callback that
   re-enters user code and contends on `session_and_task_lookup_mutex_`. One worker thread + a fixed
   10 s drain window (`:458`) + the upstream `TODO b/476205457` together make backpressure the most
   likely trigger of the `callback_thread_pool` `DEADLINE_EXCEEDED`.
2. **The CPU-sampling fallback (C) is the amplifier.** With the Metal sampler off the path, every
   token is sampled on CPU with a GPU→CPU logits copy. Invisible over 128 tokens; over tens of
   thousands it inflates per-token latency, grows the callback backlog, and pushes wall-clock toward
   both deadlines. We expect the hang to soften or move if the Metal sampler loads.
3. **The 600 s run length brushes the 10-minute whole-task ceiling (B).** Independent of A and C: a
   single turn ≥ 10 min always fails because `kDefaultTimeout` is not configurable through the Swift
   surface.

We cannot yet say which deadline fires first without LiteRT-side logging — see *Confirming
instrumentation*.

## Workarounds / hypotheses to test

- **Take the Metal sampler off the path** — run greedy with an executor-internal argmax (sampler
  unspecified) or `backend: .cpu()` and re-run the 600 s turn. If it now completes, (C) is confirmed
  as the dominant factor.
- **Embed + sign `libLiteRtTopKMetalSampler.dylib`** in the app bundle so the GPU sampler loads,
  then re-measure. Removes the CPU-sampling amplifier without changing the workload.
- **Chunk the sustained workload** into many turns each well under 10 min and short enough to drain
  callbacks (e.g. 128–512 tokens per turn, looped), instead of one 600 s turn. Avoids both A and B —
  this is our likely interim path to get a LiteRT-LM energy/throttle row.

## What would confirm this quickly

- Log which sampler path is taken (Metal GPU vs CPU fallback) at `RegisterNewSession`.
- Log `callback_thread_pool` queue depth + per-callback latency during sustained decode.
- Include **which** `WaitUntilDone` deadline fired (task-level `kDefaultTimeout` at
  `session_advanced.cc` vs the 10 s callback drain at `threaded_execution_manager.cc:458`) in the
  surfaced error.
- Consider making `kDefaultTimeout` configurable via `EngineConfig`, and resolving the
  `TODO b/476205457` so `FinishTask` no longer blocks on a synchronous callback drain.

## Impact

Blocks **sustained** (energy / throttle) measurement of LiteRT-LM on iOS GPU for Qwen3-0.6B. The
short-chat throughput / TTFT / memory rows are unaffected and reported normally. The benchmark keeps
the sustained row as a documented failure with this cause attached, rather than dropping it
([fairness rule 4](../../methodology/fairness-rules.md)). It does **not** reproduce on Gemma-4-E2B's
sustained run on the same build, which is itself a useful contrast for triage.

## Provenance

- Symptom + generated tables: [`docs/litert-lm/README.md`](README.md) (Qwen3-0.6B section, sustained
  note) and the raw JSONL under [`results/raw/`](../../results/raw/).
- Source cited above: vendored `ios/BenchmarkApp/Vendored/LiteRT-LM` @ `v0.13.1`.
