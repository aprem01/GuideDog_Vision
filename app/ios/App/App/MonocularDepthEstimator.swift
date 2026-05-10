import Foundation
import Vision
import CoreML
import CoreVideo
import UIKit

/// Monocular depth estimation for non-LiDAR iPhones.
///
/// Wraps Apple's Core ML port of Depth-Anything Small (input: 518×396 RGB,
/// output: 518×392 Grayscale16Half inverse-depth). Output values are inverse
/// depth — *higher = closer*. Values are relative until calibrated against
/// a known-distance hint (e.g. a YOLO size-based estimate).
///
/// Calibration relationship (matches what the web app does in
/// `depthToMeters` / `updateDepthCalibration`):
///     meters = k / inverseDepthValue
///     k      = modelValueAtKnownPoint * realMetersAtThatPoint
///
/// The estimator runs at most one inference at a time. If a frame arrives
/// while a previous inference is still in flight, it's dropped — fresher is
/// better than queued.
final class MonocularDepthEstimator {

    // MARK: - Public state

    public private(set) var isReady = false

    /// Latest inverse-depth pixel buffer the model produced. Float16
    /// grayscale, 518×392, *higher = closer*. Held so callers (e.g. the
    /// heatmap overlay) can sample without re-running the model.
    public private(set) var latestRawInverseDepth: CVPixelBuffer?

    /// Latest metric depth pixel buffer (Float32, meters), produced only if
    /// calibration is set. Same dimensions as the raw output. `nil` until at
    /// least one calibration hint has been supplied.
    public private(set) var latestMetricDepth: CVPixelBuffer?

    /// Whether at least one calibration hint has been applied.
    public var isCalibrated: Bool {
        stateLock.lock(); defer { stateLock.unlock() }
        return calibration != nil
    }

    // MARK: - Private state

    private var vnModel: VNCoreMLModel?
    private let processingQueue = DispatchQueue(label: "com.guidedog.monoDepth", qos: .userInitiated)
    private let stateLock = NSLock()
    private var isProcessing = false

    /// (model output value, real meters) at a calibration point — the EMA-
    /// blended pair used to convert inverse-depth → metric depth.
    private var calibration: (modelValue: Float, realMeters: Float)?

    // MARK: - Init

    public init() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.loadModel()
        }
    }

    private func loadModel() {
        guard let url = Self.findModelURL() else {
            print("MonocularDepth: DepthAnythingSmall model not found in bundle")
            return
        }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let mlModel = try MLModel(contentsOf: url, configuration: config)
            self.vnModel = try VNCoreMLModel(for: mlModel)
            self.isReady = true
            print("MonocularDepth: Depth-Anything Small loaded (\(url.lastPathComponent))")
        } catch {
            print("MonocularDepth: model load failed — \(error)")
        }
    }

    private static func findModelURL() -> URL? {
        // Compiled .mlmodelc preferred (Xcode produces this for .mlpackage
        // resources at build time).
        if let url = Bundle.main.url(forResource: "DepthAnythingSmall", withExtension: "mlmodelc") {
            return url
        }
        // Fall back to compiling the .mlpackage at runtime.
        if let pkg = Bundle.main.url(forResource: "DepthAnythingSmall", withExtension: "mlpackage") {
            do {
                return try MLModel.compileModel(at: pkg)
            } catch {
                print("MonocularDepth: runtime compile failed — \(error)")
            }
        }
        return nil
    }

    // MARK: - Public API

    /// Run depth inference on a camera frame. Drops the request if a previous
    /// inference is still in flight. completion is called on the main thread
    /// with the raw inverse-depth buffer and (if calibrated) the metric depth
    /// buffer.
    public func estimate(pixelBuffer: CVPixelBuffer,
                         completion: @escaping (_ rawInverse: CVPixelBuffer?,
                                                _ metric: CVPixelBuffer?) -> Void) {
        guard let model = vnModel else {
            completion(nil, nil); return
        }

        stateLock.lock()
        if isProcessing {
            stateLock.unlock()
            completion(nil, nil)
            return
        }
        isProcessing = true
        stateLock.unlock()

        let request = VNCoreMLRequest(model: model) { [weak self] req, _ in
            guard let self = self else { return }

            defer {
                self.stateLock.lock()
                self.isProcessing = false
                self.stateLock.unlock()
            }

            guard let result = req.results?.first as? VNPixelBufferObservation else {
                DispatchQueue.main.async { completion(nil, nil) }
                return
            }

            let raw = result.pixelBuffer
            self.latestRawInverseDepth = raw
            let metric = self.makeMetricDepth(from: raw)
            self.latestMetricDepth = metric

            DispatchQueue.main.async {
                completion(raw, metric)
            }
        }
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .right, options: [:])
        processingQueue.async {
            do { try handler.perform([request]) }
            catch {
                self.stateLock.lock()
                self.isProcessing = false
                self.stateLock.unlock()
                DispatchQueue.main.async { completion(nil, nil) }
            }
        }
    }

    /// Supply a known-distance hint to update the calibration. Typically
    /// called from NavigationEngine when YOLO detects a known-class object
    /// (person / car / etc.) whose bbox-size-based distance estimate we
    /// trust, and we sample the inverse-depth at that bbox center.
    ///
    /// EMA-blended with the existing calibration so noisy hints don't
    /// whiplash the metric conversion.
    public func updateCalibration(modelValue: Float, realMeters: Float) {
        guard modelValue > 1.0, realMeters >= 0.4, realMeters <= 8.0 else { return }
        stateLock.lock()
        defer { stateLock.unlock() }

        if let cur = calibration {
            calibration = (
                modelValue: cur.modelValue * 0.7 + modelValue * 0.3,
                realMeters: cur.realMeters * 0.7 + realMeters * 0.3
            )
        } else {
            calibration = (modelValue, realMeters)
        }
    }

    /// Sample the latest raw inverse-depth output at a normalized point
    /// (0...1 in each axis). Returns the model's value at that pixel, or
    /// `nil` if no depth has been computed yet.
    public func sampleInverseDepth(atNormalizedX nx: Float, normalizedY ny: Float) -> Float? {
        guard let buffer = latestRawInverseDepth else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        let w = CVPixelBufferGetWidth(buffer)
        let h = CVPixelBufferGetHeight(buffer)
        let bpr = CVPixelBufferGetBytesPerRow(buffer)
        guard w > 0, h > 0, let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }

        let x = min(max(Int(nx * Float(w)), 0), w - 1)
        let y = min(max(Int(ny * Float(h)), 0), h - 1)
        let ptr = base.advanced(by: y * bpr).assumingMemoryBound(to: UInt16.self)
        let bits = ptr[x]
        let f16 = Float16(bitPattern: bits)
        return Float(f16)
    }

    // MARK: - Internal: metric conversion

    /// Convert a Grayscale16Half inverse-depth pixel buffer into a Float32
    /// DepthFloat32 pixel buffer with values in metres. Requires calibration;
    /// returns nil if none has been set yet.
    private func makeMetricDepth(from inverse: CVPixelBuffer) -> CVPixelBuffer? {
        let cal: (modelValue: Float, realMeters: Float)? = {
            stateLock.lock(); defer { stateLock.unlock() }
            return calibration
        }()
        guard let cal = cal else { return nil }

        CVPixelBufferLockBaseAddress(inverse, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(inverse, .readOnly) }
        guard let inBase = CVPixelBufferGetBaseAddress(inverse) else { return nil }

        let width = CVPixelBufferGetWidth(inverse)
        let height = CVPixelBufferGetHeight(inverse)
        let inBpr = CVPixelBufferGetBytesPerRow(inverse)
        guard width > 0, height > 0 else { return nil }

        let attrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        var out: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width, height,
            kCVPixelFormatType_DepthFloat32,
            attrs as CFDictionary,
            &out
        )
        guard status == kCVReturnSuccess, let outBuffer = out else { return nil }

        CVPixelBufferLockBaseAddress(outBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(outBuffer, []) }
        guard let outBase = CVPixelBufferGetBaseAddress(outBuffer) else { return nil }

        let outBpr = CVPixelBufferGetBytesPerRow(outBuffer)
        let k: Float = cal.modelValue * cal.realMeters  // meters = k / inverseValue

        for y in 0..<height {
            let inRow  = inBase.advanced(by: y * inBpr).assumingMemoryBound(to: UInt16.self)
            let outRow = outBase.advanced(by: y * outBpr).assumingMemoryBound(to: Float32.self)
            for x in 0..<width {
                let f16 = Float16(bitPattern: inRow[x])
                let inv = Float(f16)
                // Clamp to physically sensible 0.1–10 m. Below 1.0 inverse-value
                // means "very far" — treat as 10 m sentinel.
                let meters: Float = inv > 1.0 ? min(10.0, max(0.1, k / inv)) : 10.0
                outRow[x] = meters
            }
        }

        return outBuffer
    }
}
