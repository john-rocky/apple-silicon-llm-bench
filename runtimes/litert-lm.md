# LiteRT-LM

Google's production on-device LLM runtime (the successor to the TFLite +
MediaPipe LLM Inference task).

- Reference: <https://github.com/google-ai-edge/LiteRT-LM>, Swift guide <https://ai.google.dev/edge/litert-lm/swift>
- License: Apache 2.0
- Backend: Metal GPU / CPU on Apple Silicon (also NPU on Qualcomm/Intel via platform-specific weights)
- Adapter: [`MediaPipeRuntime.swift`](../ios/BenchmarkApp/Sources/Runtimes/MediaPipeRuntime.swift) (kind `litert-lm`)

## Strengths

- Strong first-party model coverage — Gemma 4 ships in `.litertlm` first (Google's own runtime), and `litert-community` also publishes **Qwen3** (0.6B/4B), LFM/Liquid, and more. **Not** Gemma-only.
- Cross-platform parity with Android, useful for teams shipping both.
- Metal GPU acceleration on Apple Silicon; one binary covers iOS + macOS.

## Weaknesses

- Heavier dependency footprint than llama.cpp or MLX (a ~hundreds-of-MB binary xcframework).
- Model format is runtime-specific (`.litertlm`) — availability depends on Google publishing conversions.
- Swift API is **early preview** (since v0.12.0); signatures may shift between releases.

## Apple integration (wired — v0.13.1)

The official `LiteRTLM` product ships a binary xcframework with **both
`ios-arm64` and `macos-arm64` slices**, so it wires up as a plain SPM
dependency on the iOS app *and* the macOS `yardstick` CLI — no Xcode-UI-only
CocoaPod, no vendoring.

```swift
// Package.swift / project.yml
.package(url: "https://github.com/google-ai-edge/LiteRT-LM", from: "0.13.0")
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

| Model | HF repo | On-disk (variant used) |
|-------|---------|---------------------------:|
| **Qwen3 0.6B** | `litert-community/Qwen3-0.6B` | 0.50 GB (`qwen3_0_6b_mixed_int4.litertlm`) |
| Gemma 4 E2B | `litert-community/gemma-4-E2B-it-litert-lm` | 2.59 GB |
| Gemma 4 E4B | `litert-community/gemma-4-E4B-it-litert-lm` | 3.66 GB |
| Gemma 3n E2B | `google/gemma-3n-E2B-it-litert-lm` | ~3 GB |

Qwen3-0.6B is the model Lu's team is optimising; we use its **mixed blockwise-INT4**
artifact (gs32 weights + INT8 embeddings, 498 MB) so the quant lines up with the 4-bit
Qwen3-0.6B rows on MLX / CoreML / Core AI (the same repo also ships a dynamic-INT8
`Qwen3-0.6B.litertlm`, 614 MB, and a MediaTek-NPU build we skip). For the Gemma repos the
adapter prefers the standard (non-`-web`, non-NPU) `.litertlm`; each also ships `-web`
(browser-tuned) and Intel/Qualcomm NPU variants we skip for the Metal GPU comparison.

## Status

**Wired against v0.13.1.** The adapter builds against `LiteRTLM` on Mac + iOS.
The published iPhone 17 Pro decode/TTFT/memory/energy cells were **captured on
0.12.0**; re-measurement on 0.13.1 is pending — the Swift API is early-preview
and signatures may shift between releases, so rebuild on device after
`bootstrap.sh` and fix the adapter if needed. M4 Max LiteRT run still pending.
Per-device package: [`docs/litert-lm/`](../docs/litert-lm/); run commands in
[`Yardstick_USER_RUNS.md`](../../Yardstick_USER_RUNS.md).

For reference, Google's own E2B model card reports **56.5 tok/s on iPhone 17
Pro (GPU)** — a vendor figure, not a Yardstick measurement.
