// swift-tools-version:5.9
import PackageDescription

let package = Package(
  name: "litert-mac-verify",
  platforms: [.macOS(.v13)],
  dependencies: [
    .package(path: "swift-litert-lm")
  ],
  targets: [
    .executableTarget(
      name: "litert-mac-verify",
      dependencies: [
        .product(name: "LiteRTFoundation", package: "swift-litert-lm"),
        .product(name: "LiteRTLM", package: "swift-litert-lm")
      ]
    )
  ]
)
