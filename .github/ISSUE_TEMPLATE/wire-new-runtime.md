---
name: Wire a new runtime
about: Propose an adapter for a backend Yardstick doesn't have yet
title: "[runtime] add <runtime-name> adapter"
labels: ["runtime", "enhancement"]
---

## Runtime

- **Name:** (e.g. NeuralEngineKit, Onyx, …)
- **Vendor / repo:** link
- **Platforms supported:** iOS / macOS / iPadOS / visionOS
- **Model format(s):** (mlpackage / GGUF / .pte / .task / …)
- **License:** (MIT / Apache 2 / commercial)

## Sketch the adapter

The contract is `LLMRuntime` in [`ios/BenchmarkApp/Sources/Runtimes/LLMRuntime.swift`](../ios/BenchmarkApp/Sources/Runtimes/LLMRuntime.swift). At minimum we need:

- [ ] `loadModel(_:progress:)`
- [ ] `unloadModel()`
- [ ] `generate(prompt:parameters:)` streaming events
- [ ] An entry in `RuntimeKind`
- [ ] A model catalog under `ModelCatalog.<runtime>`
- [ ] At least one row in `RESULTS.md` (apples-to-apples with an existing model — same prompt + decode settings)

## Build path

How do you intend to wire the dependency?

- [ ] SPM (URL + version pin)
- [ ] Vendored xcframework (where the slices come from)
- [ ] Vendored source (where it gets cloned)
- [ ] Xcode UI add (last resort — please justify)

## Mac build

- [ ] Mac slice / target available (verified)
- [ ] iOS only (will be gated `#if canImport(...)` in the Mac CLI target)
- [ ] Unknown — please flag in this issue

## Notes

(Anything that distinguishes this runtime from what's already in the table — e.g. "first Vulkan backend", "first ANE-only path with no GPU fallback", …)
