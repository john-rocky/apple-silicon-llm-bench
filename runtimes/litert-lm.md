# LiteRT-LM

Google's production on-device LLM runtime (the successor to the TFLite +
MediaPipe LLM Inference task).

- Reference: <https://github.com/google-ai-edge/LiteRT-LM>, Swift guide <https://ai.google.dev/edge/litert-lm/swift>
- License: Apache 2.0
- Backend: Metal GPU / CPU on Apple Silicon (also NPU on Qualcomm/Intel via platform-specific weights)
- Adapter: [`MediaPipeRuntime.swift`](../ios/BenchmarkApp/Sources/Runtimes/MediaPipeRuntime.swift) (kind `litert-lm`)

## Strengths

- Strong Gemma family support — it's Google's own runtime, and Gemma 4 ships in `.litertlm` first.
- Cross-platform parity with Android, useful for teams shipping both.
- Metal GPU acceleration on Apple Silicon; one binary covers iOS + macOS.

## Weaknesses

- Heavier dependency footprint than llama.cpp or MLX (a ~hundreds-of-MB binary xcframework).
- Model format is runtime-specific (`.litertlm`) — availability depends on Google publishing conversions.
- Swift API is **early preview** (since v0.12.0); signatures may shift between releases.

## Apple integration (wired — v0.12.0)

The official `LiteRTLM` product ships a binary xcframework with **both
`ios-arm64` and `macos-arm64` slices**, so it wires up as a plain SPM
dependency on the iOS app *and* the macOS `yardstick` CLI — no Xcode-UI-only
CocoaPod, no vendoring.

```swift
// Package.swift / project.yml
.package(url: "https://github.com/google-ai-edge/LiteRT-LM", from: "0.12.0")
// product: LiteRTLM
```

- Minimum deployment: **iOS 15 / macOS 12**.
- The package declares `-Xlinker -all_load` (the LiteRT-LM static lib registers
  its CPU/GPU backends via C++ static initializers; without `-all_load` /
  `-force_load` the linker strips them and load fails at runtime). **Build
  caveat:** `-all_load` is global to the link, so if it collides with the
  vendored `llama.xcframework` / `executorch` static libs (duplicate symbols),
  switch to a scoped `-force_load <path-to-libLiteRTLM.a>` in `OTHER_LDFLAGS`.
- API surface used by the adapter: `EngineConfig(modelPath:backend:.gpu:maxNumTokens:cacheDir:)`
  → `Engine.initialize()` → `engine.createConversation(with:)` →
  `conversation.sendMessageStream(Message(prompt))`, counting streamed chunks
  as a token proxy. Prompt-token count is not surfaced in 0.12.0, so prefill
  tok/s is reported blank.

## Models targeted (`.litertlm`)

| Model | HF repo | On-disk (standard variant) |
|-------|---------|---------------------------:|
| Gemma 4 E2B | `litert-community/gemma-4-E2B-it-litert-lm` | 2.59 GB |
| Gemma 4 E4B | `litert-community/gemma-4-E4B-it-litert-lm` | 3.66 GB |
| Gemma 3n E2B | `google/gemma-3n-E2B-it-litert-lm` | ~3 GB |

The adapter prefers the standard (non-`-web`, non-NPU) `.litertlm` file. Each
repo also ships `-web` (smaller, browser-tuned) and Intel/Qualcomm NPU variants
that we skip for the Metal GPU comparison.

## Status

**Wired (v0.12.0), runs pending.** The adapter builds against `LiteRTLM` on
Mac + iOS. Decode/TTFT/memory cells in [`RESULTS.md`](../RESULTS.md) are
captured on real hardware (M4 Max + iPhone 17 Pro) — see
[`Yardstick_USER_RUNS.md`](../../Yardstick_USER_RUNS.md) for the run commands.

For reference, Google's own E2B model card reports **56.5 tok/s on iPhone 17
Pro (GPU)** — a vendor figure, not a Yardstick measurement.
