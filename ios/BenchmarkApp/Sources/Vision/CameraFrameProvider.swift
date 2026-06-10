import Foundation
import AVFoundation
import CoreVideo
#if canImport(UIKit)
import UIKit
#endif

/// Supplies camera frames to the VLM bench loop.
///
/// The runner asks for `latestFrame()` whenever the model is ready for the next
/// inference — it never queues stale frames, so "sustained FPS" is exactly
/// `inferences / second`, paced by the model, not the camera.
///
/// Two sources:
///  • `.liveCamera` — the back camera via `AVCaptureSession`. Use this for the
///     demo clip (point it at a dense, complex scene).
///  • `.loopingAsset(URL)` — a bundled video decoded on a loop. Deterministic
///     input → reproducible runs (the same scene every time, on any device).
///     Use this for the numbers that go in the repo.
public final class CameraFrameProvider: NSObject, @unchecked Sendable {
    public enum Source {
        case liveCamera
        case loopingAsset(URL)
    }

    public let source: Source

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera.frames")

    private let lock = NSLock()
    private var _latest: CVPixelBuffer?

    /// Live-camera preview layer (nil for the looping-asset source — display
    /// `latestFrame()` yourself there).
    public private(set) var previewLayer: AVCaptureVideoPreviewLayer?

    // Looping-asset decode state.
    private var assetReader: AVAssetReader?
    private var assetOutput: AVAssetReaderTrackOutput?
    private var assetTimer: DispatchSourceTimer?

    public init(source: Source) {
        self.source = source
        super.init()
    }

    /// The most recent frame, or nil before the first arrives. The runner polls
    /// this; the returned buffer is retained for the caller.
    public func latestFrame() -> CVPixelBuffer? {
        lock.lock(); defer { lock.unlock() }
        return _latest
    }

    private func setLatest(_ buffer: CVPixelBuffer) {
        lock.lock(); _latest = buffer; lock.unlock()
    }

    public func start() async throws {
        switch source {
        case .liveCamera:
            try await startLiveCamera()
        case .loopingAsset(let url):
            try startLoopingAsset(url)
        }
    }

    public func stop() {
        switch source {
        case .liveCamera:
            if session.isRunning { session.stopRunning() }
        case .loopingAsset:
            assetTimer?.cancel(); assetTimer = nil
            assetReader?.cancelReading(); assetReader = nil; assetOutput = nil
        }
    }

    // MARK: - Live camera

    private func startLiveCamera() async throws {
        // Idempotent: the view starts the session for preview, then the runner
        // re-starts the same provider — the second call is a no-op.
        if session.isRunning { return }
        guard await Self.requestCameraAccess() else {
            throw CameraError.accessDenied
        }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            throw CameraError.noCamera
        }

        session.beginConfiguration()
        session.sessionPreset = .high
        if session.canAddInput(input) { session.addInput(input) }

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }
        session.commitConfiguration()

        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        self.previewLayer = layer

        // startRunning blocks; keep it off the main thread.
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async {
                self.session.startRunning()
                cont.resume()
            }
        }
    }

    private static func requestCameraAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined:
            return await withCheckedContinuation { cont in
                AVCaptureDevice.requestAccess(for: .video) { cont.resume(returning: $0) }
            }
        default: return false
        }
    }

    // MARK: - Looping asset (deterministic)

    private func startLoopingAsset(_ url: URL) throws {
        try openReader(for: url)
        // Pull frames at ~30 fps into `_latest`; loop when the track ends.
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(33))
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            guard let output = self.assetOutput,
                  let sample = output.copyNextSampleBuffer(),
                  let pb = CMSampleBufferGetImageBuffer(sample) else {
                // End of track → rewind.
                try? self.openReader(for: url)
                return
            }
            self.setLatest(pb)
        }
        assetTimer = timer
        timer.resume()
    }

    private func openReader(for url: URL) throws {
        assetReader?.cancelReading()
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw CameraError.noVideoTrack
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        if reader.canAdd(output) { reader.add(output) }
        reader.startReading()
        assetReader = reader
        assetOutput = output
    }

    public enum CameraError: LocalizedError {
        case accessDenied, noCamera, noVideoTrack
        public var errorDescription: String? {
            switch self {
            case .accessDenied: return "Camera access denied. Enable it in Settings."
            case .noCamera: return "No back camera available."
            case .noVideoTrack: return "Reference clip has no video track."
            }
        }
    }
}

extension CameraFrameProvider: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        setLatest(pb)
    }
}
