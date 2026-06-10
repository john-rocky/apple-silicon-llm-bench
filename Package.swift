// swift-tools-version: 6.0
// Yardstick — Apple Silicon AI benchmark harness (Mac + iPhone + iPad)

import PackageDescription

let package = Package(
    name: "Yardstick",
    platforms: [
        .iOS(.v18),
        .macOS(.v14),
    ],
    products: [
        // The bench harness as a reusable library — same code drives the
        // iOS BenchmarkApp and the macOS `yardstick` CLI.
        .library(name: "YardstickKit", targets: ["YardstickKit"]),

        // macOS command-line runner.
        .executable(name: "yardstick", targets: ["YardstickCLI"]),
    ],
    dependencies: [
        // Runtime SDKs — keep this list aligned with ios/BenchmarkApp/project.yml.
        // Phase 1 wires the MLX backend only; other runtimes are added in
        // follow-up commits once their Mac toolchain is verified.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.8.1"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
        // LiteRT-LM ships a binary xcframework with ios-arm64 + macos-arm64
        // slices (≥ 0.12.0), so unlike the vendored adapters it wires up as a
        // plain SPM dependency on both targets.
        .package(url: "https://github.com/google-ai-edge/LiteRT-LM", from: "0.12.0"),
    ],
    targets: [
        .target(
            name: "YardstickKit",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
                // LiteRT-LM (`MediaPipeRuntime.swift`) — the adapter is
                // `#if canImport(LiteRTLM)`-gated, so it lights up here.
                .product(name: "LiteRTLM", package: "LiteRT-LM"),
            ],
            path: "ios/BenchmarkApp/Sources",
            // Keep Benchmark/, Models/, the MLX-related runtime sources
            // (LLMRuntime protocol, HFDownloader helper, MLXBridges, MLXRuntime)
            // and the LiteRT-LM adapter (clean SPM xcframework). The remaining
            // adapter files are excluded until their Mac build path is wired up
            // (each pulls in a different vendored SDK: AnemllCore,
            // llama.xcframework, CoreMLLLM, executorch).
            exclude: [
                "Info.plist",
                "Assets.xcassets",
                "Resources",
                "Views",
                "BenchmarkApp.swift",
                "Runtimes/CoreMLRuntime.swift",
                "Runtimes/AnemllRuntime.swift",
                "Runtimes/LlamaCppRuntime.swift",
                "Runtimes/ExecuTorchRuntime.swift",
                // Apple Core AI (CoreAILM) — iOS/macOS 27 only; the iOS app
                // wires it via XcodeGen. Excluded here like the other
                // vendor-SDK adapters so the macOS YardstickKit lib still builds.
                "Runtimes/CoreAIRuntime.swift",
            ]
        ),

        .executableTarget(
            name: "YardstickCLI",
            dependencies: ["YardstickKit"],
            path: "apple/YardstickCLI/Sources"
        ),
    ]
)
