import SwiftUI
import AVFoundation
import QuartzCore
#if canImport(UIKit)
import UIKit
#endif

/// The camera-VLM benchmark surface. Live preview behind a HUD that overlays
/// FPS, temperature, battery, ANE residency and the running caption — which is
/// exactly what you screen-record for the 20–30 s demo clip. Run it once on the
/// ANE backend and once on the GPU backend, against the same scene, to get the
/// "GPU throttles / ANE holds" pair.
struct CameraVLMView: View {
    @EnvironmentObject private var session: AppSession
    @StateObject private var controller = CameraVLMController()

    var body: some View {
        ZStack {
            CameraPreview(layer: controller.previewLayer)
                .ignoresSafeArea()
                .background(Color.black)

            VStack(spacing: 0) {
                hud
                Spacer()
                if !controller.caption.isEmpty {
                    captionCard
                }
                controls
            }
            .padding(.horizontal)
        }
        .navigationTitle("Camera VLM")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { controller.startPreview() }
        .onDisappear { controller.stop() }
    }

    // MARK: HUD

    private var hud: some View {
        VStack(spacing: 8) {
            HStack(alignment: .top) {
                placementBadge
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text(String(format: "%.1f", controller.fps))
                        .font(.system(size: 44, weight: .heavy, design: .rounded))
                        .monospacedDigit()
                    Text("FPS").font(.caption2).foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 8) {
                pill("thermometer", controller.thermalState, color: thermalColor)
                pill("bolt.fill", powerLabel, color: .yellow)
                if let ane = controller.aneResidency {
                    pill("cpu", String(format: "ANE %.0f%%", ane), color: .orange)
                }
                pill("timer", timeLabel, color: .secondary)
            }
            .font(.caption.weight(.semibold))
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding(.top, 4)
    }

    private var placementBadge: some View {
        let isANE = controller.selectedRuntime == .coreMLLLM
        return VStack(alignment: .leading, spacing: 2) {
            Text(isANE ? "ANE" : "GPU")
                .font(.headline.weight(.black))
                .foregroundStyle(.white)
                .padding(.horizontal, 10).padding(.vertical, 4)
                .background(isANE ? Color.orange : Color.purple, in: Capsule())
            Text(controller.selectedRuntime.displayName)
                .font(.caption2).foregroundStyle(.secondary)
            Text("\(controller.inferenceCount) inferences · \(controller.tokens) tok")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private var captionCard: some View {
        Text(controller.caption)
            .font(.callout)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            .padding(.bottom, 8)
    }

    // MARK: Controls

    private var controls: some View {
        VStack(spacing: 10) {
            if !controller.isRunning {
                Picker("Backend", selection: $controller.selectedRuntime) {
                    ForEach(AppSession.vlmRuntimeKinds) { kind in
                        Text(kind == .coreMLLLM ? "CoreML / ANE" : "MLX / GPU").tag(kind)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: controller.selectedRuntime) { _, kind in
                    controller.reconcileModel(session: session)
                }

                HStack {
                    Picker("Model", selection: $controller.selectedModel) {
                        ForEach(session.vlmRuntime(for: controller.selectedRuntime).supportedModels, id: \.id) { m in
                            Text(m.displayName).tag(m)
                        }
                    }
                    .pickerStyle(.menu)
                    Spacer()
                    Picker("Duration", selection: $controller.durationSeconds) {
                        Text("30 s clip").tag(TimeInterval(30))
                        Text("2 min").tag(TimeInterval(120))
                        Text("10 min").tag(TimeInterval(600))
                    }
                    .pickerStyle(.menu)
                }
                .font(.subheadline)
            }

            if let err = controller.errorText {
                Text(err).font(.caption).foregroundStyle(.red)
            }
            if case .loadingModel(let p) = controller.phase {
                ProgressView(value: p) { Text("Loading model… \(Int(p * 100))%").font(.caption) }
            }
            if let result = controller.lastResult, !controller.isRunning {
                resultSummary(result)
            }

            Button(action: { controller.toggle(session: session) }) {
                Label(controller.isRunning ? "Stop" : "Start session",
                      systemImage: controller.isRunning ? "stop.circle.fill" : "record.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(controller.isRunning ? .red : .accentColor)
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding(.bottom, 8)
    }

    private func resultSummary(_ r: VLMRunResult) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Done — \(r.placement.uppercased()) · \(r.metrics.inferenceCount) inferences")
                .font(.caption.weight(.semibold))
            Text(String(format: "sustained %.2f FPS · start %.2f → end %.2f (%.0f%% drop)",
                        r.metrics.sustainedFPS, r.metrics.startFPS, r.metrics.endFPS, r.metrics.fpsDropPercent))
                .font(.caption2).foregroundStyle(.secondary)
            Text(String(format: "TTFT p50 %.0f ms · peak thermal %@%@",
                        r.metrics.ttftMedianMS, r.metrics.peakThermalState,
                        r.metrics.averagePackagePowerW.map { String(format: " · %.1f W", $0) } ?? ""))
                .font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: Bits

    private func pill(_ icon: String, _ text: String, color: Color) -> some View {
        Label(text, systemImage: icon)
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color == .secondary ? Color.primary : color)
    }

    private var thermalColor: Color {
        switch controller.thermalState {
        case "nominal": return .green
        case "fair": return .yellow
        case "serious": return .orange
        default: return .red
        }
    }

    private var powerLabel: String {
        if let w = controller.lastResult?.metrics.averagePackagePowerW { return String(format: "%.1f W", w) }
        return "—"
    }

    private var timeLabel: String {
        guard controller.isRunning else { return "ready" }
        let r = Int(controller.remaining)
        return String(format: "%d:%02d left", r / 60, r % 60)
    }
}

// MARK: - Controller

@MainActor
final class CameraVLMController: ObservableObject {
    @Published var phase: CameraVLMRunner.Phase = .idle
    @Published var fps: Double = 0
    @Published var inferenceCount = 0
    @Published var tokens = 0
    @Published var caption = ""
    @Published var thermalState = "nominal"
    @Published var aneResidency: Double?
    @Published var elapsed: TimeInterval = 0
    @Published var remaining: TimeInterval = 0
    @Published var lastResult: VLMRunResult?
    @Published var errorText: String?
    @Published var previewLayer: CALayer?

    @Published var selectedRuntime: RuntimeKind = .mlxSwift
    @Published var selectedModel: ModelInfo = VLMModelCatalog.mlx[0]
    @Published var durationSeconds: TimeInterval = 600

    private var provider: CameraFrameProvider?
    private let runner = CameraVLMRunner()
    private var runTask: Task<Void, Never>?

    var isRunning: Bool {
        switch phase {
        case .idle, .done, .failed: return false
        default: return true
        }
    }

    func reconcileModel(session: AppSession) {
        let models = session.vlmRuntime(for: selectedRuntime).supportedModels
        if !models.contains(where: { $0.id == selectedModel.id }), let first = models.first {
            selectedModel = first
        }
    }

    func startPreview() {
        guard provider == nil else { return }
        let p = CameraFrameProvider(source: .liveCamera)
        provider = p
        Task {
            do {
                try await p.start()
                self.previewLayer = p.previewLayer
            } catch {
                self.errorText = error.localizedDescription
            }
        }
    }

    func toggle(session: AppSession) {
        if isRunning { stop() } else { run(session: session) }
    }

    func run(session: AppSession) {
        errorText = nil
        lastResult = nil
        caption = ""
        let runtime = session.vlmRuntime(for: selectedRuntime)
        let model = selectedModel
        let task = CameraVLMTask(durationSeconds: durationSeconds)
        let p = provider ?? CameraFrameProvider(source: .liveCamera)
        provider = p
        phase = .loadingModel(progress: 0)

        runTask = Task {
            let stream = await runner.snapshots()
            let observer = Task {
                for await s in stream { await MainActor.run { self.apply(s) } }
            }
            defer { observer.cancel() }
            do {
                let cold = (await runtime.loadedModelId) != model.id
                let result = try await runner.run(.init(
                    runtime: runtime, model: model, task: task, frameProvider: p, coldRun: cold
                ))
                let url = try? VLMResultStore.save(result)
                await MainActor.run {
                    self.lastResult = result
                    self.phase = .done
                    if let url { print("[CameraVLM] saved \(url.lastPathComponent)") }
                }
                // Re-arm the preview for another take.
                await MainActor.run { self.previewLayer = p.previewLayer }
            } catch is CancellationError {
                await MainActor.run { self.phase = .idle }
            } catch {
                await MainActor.run {
                    self.errorText = error.localizedDescription
                    self.phase = .failed(error.localizedDescription)
                }
            }
        }
    }

    func stop() {
        runTask?.cancel()
        Task { await runner.cancel() }
    }

    private func apply(_ s: CameraVLMRunner.Snapshot) {
        phase = s.phase
        fps = s.currentFPS
        inferenceCount = s.inferenceCount
        tokens = s.tokens
        if !s.lastCaption.isEmpty { caption = s.lastCaption }
        thermalState = s.thermalState
        if let a = s.aneResidencyPercent { aneResidency = a }
        elapsed = s.elapsed
        remaining = s.remaining
    }
}

// MARK: - Preview layer host

private struct CameraPreview: UIViewRepresentable {
    let layer: CALayer?

    func makeUIView(context: Context) -> PreviewHostView { PreviewHostView() }
    func updateUIView(_ view: PreviewHostView, context: Context) { view.attach(layer) }
}

private final class PreviewHostView: UIView {
    private var attached: CALayer?

    func attach(_ newLayer: CALayer?) {
        guard attached !== newLayer else { return }
        attached?.removeFromSuperlayer()
        if let newLayer { layer.addSublayer(newLayer) }
        attached = newLayer
        setNeedsLayout()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        attached?.frame = bounds
    }
}
