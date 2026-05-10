import ARKit
import AVFoundation
import CoreVideo
import Foundation
import QuartzCore

/// Protocol for NavigationEngine to communicate events to the Expo module
protocol NavigationEventDelegate: AnyObject {
    func navigationDidUpdate(distances: [String: Any], risk: [String: Any], detections: [[String: Any]])
    func navigationDidDetect(label: String, direction: String, distance: Float?)
}

/// Core navigation engine — manages AR session, depth, object detection, and feedback
class NavigationEngine: NSObject, ARSessionDelegate, VoiceCommandDelegate {

    // MARK: - Properties

    private var arSession: ARSession?
    private let detector = ObjectDetector()      // Vision + YOLOv8n CoreML
    private let meshClassifier = MeshClassifier() // ARKit mesh (walls, doors, floors)
    private let sceneSegmenter = SceneSegmenter() // DeepLabV3 semantic segmentation
    private let sceneDescriber = AISceneDescriber()
    private var speech: SpeechController?
    private var spatialAudio: SpatialAudioController?
    private var haptics: HapticController?

    weak var delegate: NavigationEventDelegate?

    // JS bridge callbacks — set by ViewController before calling start()
    var onDepthUpdate: ((Float, Float, Float) -> Void)?   // (center, left, right) in metres
    var onCameraFrame: ((CVPixelBuffer) -> Void)?          // ~1/sec for JS preview & AI
    var onDetectionEvent: ((String, String) -> Void)?      // (label, direction)
    var onDepthMap: ((CVPixelBuffer) -> Void)?             // raw depth buffer for the demo heatmap overlay

    var isRunning = false
    private(set) var hasLiDAR = false

    private var frameCount = 0
    private var lastDebugLog: TimeInterval = 0

    // Guards to prevent concurrent Vision/CoreML requests from piling up
    // (causes ANE overload → camera pipeline stall → frame freeze)
    private var isDetecting = false
    private var isSegmenting = false

    // Progressive distance bands with hysteresis
    // 0 = safe, 1 = approaching (3.0m), 2 = caution (2.0m), 3 = danger (1.0m), 4 = critical (0.4m)
    private var centerBand: Int = 0
    private var lastCriticalSpeak: TimeInterval = 0  // for critical re-announce every 1.5s

    // Hysteresis-aware risk tracking
    private var currentRiskL: RiskLevel = .safe
    private var currentRiskC: RiskLevel = .safe
    private var currentRiskR: RiskLevel = .safe
    private var sLeft: Float?
    private var sCenter: Float?
    private var sRight: Float?
    private var lastDepthMap: CVPixelBuffer?  // cached for per-object distance sampling

    // Latest detections for getDetections() call
    private var latestDetections: [[String: Any]] = []

    private let detectionQueue = DispatchQueue(label: "com.blindguide.detection", qos: .userInitiated)

    // AR breadcrumb — "remember this spot" / "take me back"
    private var savedAnchor: ARAnchor?
    private var guideBackTimer: Timer?
    private var lastGuideAnnouncement: (distance: Float, time: TimeInterval) = (0, 0)

    // Monocular depth fallback for non-LiDAR phones. Runs Depth-Anything Small
    // (~3 FPS) and feeds metric depth into the same processDepth() pipeline
    // LiDAR uses, once calibrated against a YOLO size-based distance hint.
    private lazy var monoDepth: MonocularDepthEstimator = MonocularDepthEstimator()
    private var monoDepthInflight = false

    // MARK: - Init

    override init() {
        super.init()
        print("NavigationEngine: init starting")

        // Check LiDAR support safely
        if ARWorldTrackingConfiguration.isSupported {
            hasLiDAR = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        }
        print("NavigationEngine: LiDAR=\(hasLiDAR), ARSupported=\(ARWorldTrackingConfiguration.isSupported)")
    }

    // MARK: - Public Methods

    func start() {
        print("NavigationEngine: start() called")

        UIApplication.shared.isIdleTimerDisabled = true
        configureAudioSession()

        // Speech first — must exist immediately for welcome message
        if speech == nil { speech = SpeechController() }
        isRunning = true
        speech?.speak("Starting.", urgency: 7.0)

        // Callbacks (lightweight, no blocking)
        sceneDescriber.onDescription = { [weak self] description, provider in
            print("AISceneDescriber: [\(provider)] \(description)")
            let short = String(description.prefix(80)).lowercased()
            if short == self?.lastCloudResult { return }
            self?.lastCloudResult = short
            let userInitiated = self?.isUserScan ?? false
            self?.isUserScan = false
            self?.speech?.speak(description, urgency: userInitiated ? 7.0 : 1.0)
        }
        sceneDescriber.onError = { [weak self] error in
            print("AISceneDescriber: Error — \(error)")
            self?.describeAreaLocal()
        }

        // Defer heavy init — lets UI render + speech play before blocking
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            guard let self = self else { return }

            if self.haptics == nil { self.haptics = HapticController() }
            // SpatialAudio delayed further — its AVAudioEngine conflicts with speech if created too early
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
                if self?.spatialAudio == nil {
                    let sa = SpatialAudioController()
                    sa.speechController = self?.speech
                    self?.spatialAudio = sa
                }
            }

            guard ARWorldTrackingConfiguration.isSupported else {
                print("⚠️ AR not supported — limited mode")
                return
            }

            if self.arSession == nil {
                self.arSession = ARSession()
                self.arSession?.delegate = self
            }

            let config = ARWorldTrackingConfiguration()
            if self.hasLiDAR {
                config.frameSemantics.insert(.sceneDepth)
                if ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
                    config.sceneReconstruction = .meshWithClassification
                }
            }

            self.arSession?.run(config, options: [.resetTracking, .removeExistingAnchors])
            print("✅ NavigationEngine started (LiDAR: \(self.hasLiDAR))")
        }
    }

    private func configureAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            // playAndRecord allows speech output + mic input
            // allowBluetoothA2DP routes audio to AirPods/Bluetooth headphones
            // mixWithOthers allows speech to play alongside AR audio
            // duckOthers lowers other audio when speaking
            try audioSession.setCategory(
                .playAndRecord,
                mode: .spokenAudio,
                options: [.allowBluetoothA2DP, .mixWithOthers, .duckOthers]
            )
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)

            // Route to Bluetooth if available, otherwise speaker
            let currentRoute = audioSession.currentRoute
            let hasBluetooth = currentRoute.outputs.contains { $0.portType == .bluetoothA2DP || $0.portType == .bluetoothHFP || $0.portType == .bluetoothLE }
            if !hasBluetooth {
                try audioSession.overrideOutputAudioPort(.speaker)
            }
            print("NavigationEngine: Audio session configured (Bluetooth: \(hasBluetooth))")
        } catch {
            print("NavigationEngine: Audio session error: \(error)")
        }
    }

    func stop() {
        arSession?.pause()
        isRunning = false
        haptics?.stop()
        spatialAudio?.stopAll()
        cancelGuideBackTimer()
        print("🛑 NavigationEngine stopped")
    }

    func getLatestDetections() -> [[String: Any]] {
        return latestDetections
    }

    /// Expose the AR session for camera preview
    func getARSession() -> ARSession? {
        return arSession
    }

    // MARK: - AR Session Delegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isRunning else { return }
        frameCount += 1

        // LAYER 1: Depth (every 4th frame, LiDAR only)
        // Skip when SLAM tracking is poor — depth readings are unreliable during initialization
        let trackingGood: Bool
        switch frame.camera.trackingState {
        case .normal: trackingGood = true
        default: trackingGood = false
        }
        if hasLiDAR, frameCount % 4 == 0, trackingGood,
           let depthMap = frame.sceneDepth?.depthMap {
            // Process depth on background thread to avoid blocking AR frame delivery
            detectionQueue.async { [weak self] in
                self?.processDepth(depthMap)
            }
        }

        // LAYER 2: Object detection — Vision + YOLOv8n CoreML (~1.5 FPS)
        // isDetecting guard: skip if prior Vision request still running (prevents ANE pileup)
        if frameCount % 20 == 0, detector.isReady, !isDetecting {
            isDetecting = true
            detector.detect(pixelBuffer: frame.capturedImage)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { [weak self] in
                guard let self = self else { return }
                let objects = self.detector.detectedObjects

                // Ghost filter: track which objects persist across frames
                var currentLabels = Set<String>()
                for obj in objects {
                    let label = obj.label.lowercased()
                    currentLabels.insert(label)
                    self.detectionStreak[label] = (self.detectionStreak[label] ?? 0) + 1
                }
                // Reset streak for objects NOT seen this frame
                for key in self.detectionStreak.keys {
                    if !currentLabels.contains(key) { self.detectionStreak[key] = 0 }
                }

                // Only pass objects with 2+ consecutive detections
                let confirmed = objects.filter { self.detectionStreak[$0.label.lowercased()] ?? 0 >= 2 }
                if !confirmed.isEmpty { self.processDetections(confirmed) }

                self.isDetecting = false
            }
        }

        // LAYER 3: Mesh classification — walls/doors/floors via ARKit (~2 FPS)
        if frameCount % 15 == 0, hasLiDAR {
            let f = frame
            detectionQueue.async { [weak self] in
                guard let self = self else { return }
                self.meshClassifier.classify(frame: f)
                self.processMeshHits()
            }
        }

        // LAYER 4: Scene segmentation — DeepLabV3 furniture/people (~0.5 FPS)
        // isSegmenting guard: DeepLabV3 is the heaviest model — never run concurrently
        if frameCount % 60 == 0, sceneSegmenter.isReady, !isSegmenting {
            isSegmenting = true
            sceneSegmenter.segment(pixelBuffer: frame.capturedImage)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
                self?.processSegmentation()
                self?.isSegmenting = false
            }
        }

        // LAYER 5: Camera frame for native UIImageView preview (~10fps at 30fps AR)
        if frameCount % 3 == 0 {
            onCameraFrame?(frame.capturedImage)
        }

        // LAYER 6: Monocular depth fallback (non-LiDAR phones only, ~3 FPS).
        // Runs Depth-Anything Small on the camera frame. If we've calibrated
        // against a YOLO known-height detection, the metric output feeds into
        // the same processDepth() the LiDAR path uses — same alerts, same
        // haptics, same spatial audio.
        if !hasLiDAR, frameCount % 20 == 0,
           monoDepth.isReady, !monoDepthInflight, trackingGood {
            monoDepthInflight = true
            let f = frame
            monoDepth.estimate(pixelBuffer: f.capturedImage) { [weak self] _, metric in
                guard let self = self else { return }
                self.monoDepthInflight = false
                if let metric = metric {
                    // Same downstream pipeline as LiDAR: risk bands, haptics,
                    // spatial audio, JS bridge, the entire cascade.
                    self.detectionQueue.async { self.processDepth(metric) }
                    // Demo heatmap overlay works too — DepthOverlayView only
                    // cares that the buffer is Float32 metric depth.
                    self.onDepthMap?(metric)
                }
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("❌ ARSession failed: \(error.localizedDescription)")
        isRunning = false
    }

    func sessionWasInterrupted(_ session: ARSession) {
        print("⚠️ ARSession interrupted — pausing feedback")
        // Reset in-flight guards so they don't stay stuck if interrupted mid-request
        isDetecting = false
        isSegmenting = false
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        print("✅ ARSession interruption ended — resuming")
        // Reset band tracking so we don't fire stale alerts on resume
        centerBand = 0
        sLeft = nil; sCenter = nil; sRight = nil
        currentRiskL = .safe; currentRiskC = .safe; currentRiskR = .safe
        // World coordinates may have shifted; the saved breadcrumb is no longer
        // trustworthy. Drop it and ask the user to re-mark.
        if savedAnchor != nil {
            savedAnchor = nil
            cancelGuideBackTimer()
            speech?.speak("Lost the saved spot after the camera reset. Mark it again.", urgency: 6.0)
        }
    }

    // MARK: - Depth Processing

    private func processDepth(_ depthMap: CVPixelBuffer) {
        lastDepthMap = depthMap

        // Forward to the demo heatmap overlay (callback handles its own locking).
        // Called before our read-lock since CVPixelBuffer locks are reentrant
        // for read-only but it's cleaner to keep them disjoint.
        onDepthMap?(depthMap)

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        guard let baseAddr = CVPixelBufferGetBaseAddress(depthMap) else { return }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        guard width > 0, height > 0 else { return }

        let base = baseAddr.assumingMemoryBound(to: Float32.self)

        // Scan top 25%–65% of frame (excludes floor in bottom 35% and sky/ceiling in top 25%)
        // This is the key fix for L/R undershooting — floor readings were pulling distances down
        let startY = Int(Double(height) * 0.25)
        let endY = Int(Double(height) * 0.65)
        let col1 = width / 3
        let col2 = (width / 3) * 2
        let step = 6

        var lefts: [Float] = []
        var centers: [Float] = []
        var rights: [Float] = []

        for y in stride(from: startY, to: endY, by: step) {
            for x in stride(from: 0, to: width, by: step) {
                let idx = y * width + x
                guard idx >= 0, idx < width * height else { continue }
                let d = base[idx]
                if d < 0.03 || d > 5.0 { continue }  // 3cm min (was 10cm)
                if x < col1 { lefts.append(d) }
                else if x < col2 { centers.append(d) }
                else { rights.append(d) }
            }
        }

        let alpha: Float = 0.4  // faster response (was 0.2)
        // Center: 20th percentile (biased toward closest obstacle — what you'll walk into)
        // Left/Right: 50th percentile (median — avoids floor/edge noise, gives true side distance)
        sLeft = smooth(sLeft, getMedian(lefts), alpha)
        sCenter = smooth(sCenter, getPercentile(centers), alpha)
        sRight = smooth(sRight, getMedian(rights), alpha)

        // Hysteresis-aware risk — prevents oscillation at boundaries
        currentRiskL = RiskSolver.analyze(distance: sLeft,   current: currentRiskL)
        currentRiskC = RiskSolver.analyze(distance: sCenter, current: currentRiskC)
        currentRiskR = RiskSolver.analyze(distance: sRight,  current: currentRiskR)

        let riskL = currentRiskL
        let riskC = currentRiskC
        let riskR = currentRiskR

        // Debug log at 2Hz
        let now = CACurrentMediaTime()
        if now - lastDebugLog > 0.5 {
            lastDebugLog = now
            let lStr = sLeft.map  { String(format: "%.2fm", $0) } ?? "---"
            let cStr = sCenter.map { String(format: "%.2fm", $0) } ?? "---"
            let rStr = sRight.map { String(format: "%.2fm", $0) } ?? "---"
            #if DEBUG
            print("📏 LIDAR  L:\(lStr)[\(riskString(riskL))]  C:\(cStr)[\(riskString(riskC))]  R:\(rStr)[\(riskString(riskR))]")
            #endif
        }

        // ── PROGRESSIVE DISTANCE BANDS ────────────────────────────────────────────
        // Each band fires ONCE when first entered. Haptic fires immediately (no latency),
        // beep fires at 50ms, speech fires at 100ms so user gets spatial cue before words.
        // Critical band repeats every 1.5s since the user genuinely needs to stop.
        let c = sCenter ?? 5.0
        let newBand: Int
        if c < 0.4      { newBand = 4 }  // critical — almost touching
        else if c < 1.0 { newBand = 3 }  // danger
        else if c < 2.0 { newBand = 2 }  // caution
        else if c < 3.0 { newBand = 1 }  // approaching
        else            { newBand = 0 }  // safe

        let bandEntered = newBand > centerBand    // escalation
        let criticalRepeat = newBand == 4 && (now - lastCriticalSpeak) > 3.0

        if bandEntered || criticalRepeat {
            centerBand = newBand

            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.haptics?.updateFeedback(risk: riskC, distance: c)

                // Don't interrupt active scan with distance band speech
                guard !self.isScanActive else { return }

                switch newBand {
                case 4: self.speech?.speak("Stop", urgency: 5.0)
                case 3: self.speech?.speak("Stop, something close", urgency: 5.0)
                case 2: self.speech?.speak("Heads up", urgency: 4.0)
                case 1: self.speech?.speak("Something ahead", urgency: 3.0)
                default: break
                }
            }
            if newBand == 4 { lastCriticalSpeak = now }
        }

        // Reset band when user moves away (hysteresis: clear band only well past threshold)
        if newBand < centerBand {
            // Only drop a band if clearly past the exit threshold
            let clearThreshold: Float
            switch centerBand {
            case 4: clearThreshold = 0.6
            case 3: clearThreshold = 1.3
            case 2: clearThreshold = 2.4
            case 1: clearThreshold = 3.3
            default: clearThreshold = 0
            }
            if c > clearThreshold { centerBand = newBand }
        }

        // All feedback on main thread
        let leftVal = sLeft, centerVal = sCenter, rightVal = sRight
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }

            self.haptics?.updateFeedback(risk: riskC, distance: centerVal ?? 5.0)
            self.spatialAudio?.updateFromDepth(
                leftDist: leftVal, centerDist: centerVal, rightDist: rightVal,
                leftRisk: riskL, centerRisk: riskC, rightRisk: riskR
            )

            // Bridge to JS
            self.onDepthUpdate?(centerVal ?? 5.0, leftVal ?? 5.0, rightVal ?? 5.0)

            // Emit event to delegate
            let distances: [String: Any] = [
                "left": leftVal as Any,
                "center": centerVal as Any,
                "right": rightVal as Any
            ]
            let risk: [String: Any] = [
                "left": self.riskString(riskL),
                "center": self.riskString(riskC),
                "right": self.riskString(riskR)
            ]
            self.delegate?.navigationDidUpdate(distances: distances, risk: risk, detections: self.latestDetections)
        }
    }

    // MARK: - Per-Object Depth Sampling

    /// Sample the LiDAR depth map at a specific bounding box region.
    /// Returns the 20th percentile depth (meters) within the box, or nil if no depth map.
    private func sampleDepthAt(box: CGRect) -> Float? {
        guard let depthMap = lastDepthMap else { return nil }

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        guard let baseAddr = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
        let w = CVPixelBufferGetWidth(depthMap)
        let h = CVPixelBufferGetHeight(depthMap)
        guard w > 0, h > 0 else { return nil }
        let base = baseAddr.assumingMemoryBound(to: Float32.self)

        // Map bounding box (0-1 normalized) to depth map pixels
        let xStart = max(0, Int(Float(box.minX) * Float(w)))
        let xEnd   = min(w, Int(Float(box.maxX) * Float(w)))
        let yStart = max(0, Int(Float(box.minY) * Float(h)))
        let yEnd   = min(h, Int(Float(box.maxY) * Float(h)))

        var samples: [Float] = []
        let step = 4  // sample every 4th pixel for speed
        for y in stride(from: yStart, to: yEnd, by: step) {
            for x in stride(from: xStart, to: xEnd, by: step) {
                let d = base[y * w + x]
                if d > 0.03 && d < 5.0 { samples.append(d) }
            }
        }

        guard !samples.isEmpty else { return nil }
        let sorted = samples.sorted()
        return sorted[min(Int(Float(sorted.count) * 0.2), sorted.count - 1)]
    }

    // MARK: - Size-Based Distance Estimation

    // Known real-world heights in meters for YOLO classes
    private static let knownHeights: [String: Float] = [
        "person": 1.7, "bicycle": 1.0, "car": 1.5, "motorcycle": 1.1,
        "bus": 3.0, "truck": 3.0, "chair": 0.9, "bench": 0.85,
        "couch": 0.85, "bed": 0.6, "dining table": 0.75,
        "refrigerator": 1.7, "fire hydrant": 0.6, "stop sign": 2.1,
        "parking meter": 1.2, "potted plant": 0.5, "backpack": 0.5,
        "laptop": 0.3, "tv": 0.5
    ]

    /// Estimate distance using object's bounding box height + camera focal length.
    /// Returns meters, or nil if object class has no known height.
    private func estimateDistanceFromSize(_ det: DetectedObject) -> Float? {
        let label = det.label.lowercased()
        guard let realHeight = NavigationEngine.knownHeights[label] else { return nil }

        // Get focal length from ARKit camera intrinsics (pixels)
        guard let frame = arSession?.currentFrame else { return nil }
        let fy = frame.camera.intrinsics[1][1]  // vertical focal length in pixels

        // Bounding box height is normalized (0-1), convert to pixels
        let imageHeight = Float(frame.camera.imageResolution.height)
        let bboxHeightPx = Float(det.boundingBox.height) * imageHeight

        guard bboxHeightPx > 10 else { return nil }  // too small, unreliable

        // distance = (realHeight * focalLength) / bboxHeightPixels
        let dist = (realHeight * fy) / bboxHeightPx
        return max(0.3, min(dist, 15.0))  // clamp 0.3m – 15m
    }

    // MARK: - Object Detection

    // Priority tiers — tier 1 announced first, interrupts lower tiers
    private static let tier1: Set<String> = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]
    private static let tier2: Set<String> = ["chair", "bench", "dining table", "couch"]
    private static let tier3: Set<String> = ["fire hydrant", "stop sign", "parking meter",
                                             "refrigerator", "bed", "backpack",
                                             "laptop", "tv"]
    private static let allRelevant = tier1.union(tier2).union(tier3)

    private static func tierOf(_ label: String) -> Int {
        if tier1.contains(label) { return 1 }
        if tier2.contains(label) { return 2 }
        return 3
    }

    private var lastAnnounced: [String: (distance: Float, time: TimeInterval)] = [:]
    private var lastTier1Speech: TimeInterval = 0

    // Ghost filter: track consecutive detections per object class
    private var detectionStreak: [String: Int] = [:]

    // Object tracking: track position + distance over time for approach detection
    private var trackedObjects: [String: (dist: Float, x: Float, time: TimeInterval)] = [:]

    // Smart cloud AI — triggered by scene changes, not timer
    private var lastCloudScan: TimeInterval = 0
    private var lastMeshLabel: String = ""
    private var lastCloudResult: String = ""
    private let cloudMinInterval: TimeInterval = 20.0
    private var isUserScan = false
    private var isScanActive = false  // suppresses distance band speech during scan

    private func processDetections(_ results: [DetectedObject]) {
        var eventData: [[String: Any]] = []
        let now = CACurrentMediaTime()

        // Build ranked list: filter → add distance → sort by tier then distance
        struct Ranked {
            let det: DetectedObject; let label: String
            let direction: String; let dist: Float; let tier: Int
        }
        var ranked: [Ranked] = []

        for det in results {
            let label = det.label.lowercased()
            guard NavigationEngine.allRelevant.contains(label) else { continue }

            let boxX = Float(det.boundingBox.midX)
            let direction = boxX < 0.33 ? "left" : boxX > 0.67 ? "right" : "ahead"

            // Distance: prefer size-based estimate (per-object), fallback to LiDAR zone
            let sizeDist = estimateDistanceFromSize(det)
            let zoneDist: Float?
            switch direction {
            case "left":  zoneDist = sLeft
            case "right": zoneDist = sRight
            default:      zoneDist = sCenter
            }

            // Use size-based if available, cross-check with LiDAR zone
            let dist: Float
            if let sd = sizeDist {
                if let zd = zoneDist, abs(sd - zd) < 1.0 {
                    // Both agree within 3ft — average them for best accuracy
                    dist = (sd + zd) / 2.0
                } else {
                    // Disagree — trust size-based (it's per-object, not zone average)
                    dist = sd
                }
            } else {
                dist = zoneDist ?? 5.0
            }

            // Calibrate the monocular depth estimator on non-LiDAR phones:
            // when we have a confident size-based distance for a known-height
            // object, sample the inverse-depth at the bbox center and feed it
            // in. After a few such hints, the metric depth output becomes
            // trustworthy and the LiDAR-style alert pipeline takes over.
            if !hasLiDAR, monoDepth.isReady,
               let sizeDist = sizeDist,
               det.confidence >= 0.7,
               sizeDist >= 0.5, sizeDist <= 5.0 {
                let nx = Float(det.boundingBox.midX)
                // Vision bounding boxes use bottom-left origin; the depth
                // buffer is top-left. Flip Y so we sample the right point.
                let ny = 1.0 - Float(det.boundingBox.midY)
                if let inv = monoDepth.sampleInverseDepth(atNormalizedX: nx, normalizedY: ny) {
                    monoDepth.updateCalibration(modelValue: inv, realMeters: sizeDist)
                }
            }

            eventData.append(["label": det.label, "confidence": det.confidence, "direction": direction])
            ranked.append(Ranked(det: det, label: label, direction: direction,
                                 dist: dist, tier: NavigationEngine.tierOf(label)))
        }

        // Tier 1 first, then closest first
        ranked.sort { $0.tier != $1.tier ? $0.tier < $1.tier : $0.dist < $1.dist }

        // Object tracking: keep position history per (label, direction) so the
        // standard re-announcement check below has stable distance baselines.
        // Fast-approach interjections were removed — they fired on every cycle
        // for moving people/cars and felt like the app was nagging.
        var currentTracked = Set<String>()
        for r in ranked {
            let key = "\(r.label)_\(r.direction)"
            currentTracked.insert(key)
            trackedObjects[key] = (dist: r.dist, x: Float(r.det.boundingBox.midX), time: now)
        }
        // Clean up objects not seen
        for key in trackedObjects.keys {
            if !currentTracked.contains(key) { trackedObjects.removeValue(forKey: key) }
        }

        // Find the single most important object to announce
        // (1 per cycle — prevents speech queue buildup and stale announcements)
        var bestToSpeak: Ranked? = nil

        // Global tier-1 self-throttle: when several people/cars are in view,
        // we don't want a flood of "person ahead, person ahead" overlapping.
        // The previous announcement needs ~2.5s to finish; gate the next.
        let tier1Throttled = (now - lastTier1Speech) < 2.5

        for r in ranked {
            // Tier gating
            if r.tier == 2 && now - lastTier1Speech < 3.0 { continue }
            if r.tier == 3 && now - lastTier1Speech < 5.0 { continue }
            if r.tier == 1 && tier1Throttled { continue }

            let key = "\(r.label)_\(r.direction)"

            // Repetition check
            var shouldSpeak = false
            if let prev = lastAnnounced[key] {
                let dt = now - prev.time
                if r.tier == 1 {
                    // Tier 1 (people, vehicles): only re-announce when approaching
                    // meaningfully closer, OR after a long pause if still in view.
                    // This prevents "stuck on people" when the user walks past
                    // multiple people who all key to person_<direction>.
                    let approachingCloser = r.dist < prev.distance - 0.5
                    if approachingCloser && dt > 3.0 { shouldSpeak = true }
                    else if dt > 8.0 { shouldSpeak = true }
                } else {
                    let dd = abs(r.dist - prev.distance)
                    if dd > 1.0 && dt > 4.0 { shouldSpeak = true }
                    else if dt > 15.0 { shouldSpeak = true }
                }
            } else {
                shouldSpeak = true
            }

            if shouldSpeak { bestToSpeak = r; break }  // take the first (highest priority)
        }

        // Speak the single most important detection with SHORT text
        // (SpeechController's priority system handles interruption — no manual cancel needed)
        if let r = bestToSpeak {
            let key = "\(r.label)_\(r.direction)"
            let feet = Int(r.dist * 3.28084)
            let conf = r.det.confidence
            let name = conf >= 0.7 ? r.det.label.capitalized : "Something"

            let text: String
            let urgency: Double

            if r.dist < 0.5 {
                text = "Stop. \(name) \(r.direction)"
                urgency = 5.0
            } else if r.dist < 1.0 {
                text = "\(name) close \(r.direction)"
                urgency = 5.0
            } else {
                text = "\(name), \(feet) feet"
                urgency = r.tier == 1 ? 4.0 : 3.0
            }

            speech?.speak(text, urgency: urgency)
            lastAnnounced[key] = (distance: r.dist, time: now)
            if r.tier == 1 { lastTier1Speech = now }

            let dir = r.direction; let lbl = r.det.label; let zd = r.dist
            DispatchQueue.main.async { [weak self] in
                self?.onDetectionEvent?(lbl, dir)
                self?.delegate?.navigationDidDetect(label: lbl, direction: dir, distance: zd)
            }
        }

        latestDetections = eventData
        // Drop stale tracking entries after 10s so a returning object is treated
        // as fresh (rather than gated by the 8s tier-1 cooldown above).
        let staleKeys = lastAnnounced.filter { now - $0.value.time > 10 }.map { $0.key }
        for key in staleKeys { lastAnnounced.removeValue(forKey: key) }
    }

    // MARK: - Mesh Classification (walls, doors, floors from ARKit)

    private func processMeshHits() {
        guard let hit = meshClassifier.centerHit else { return }
        let now = CACurrentMediaTime()

        let dir = hit.direction == "center" ? "" : " \(hit.direction)"
        let feet = String(format: "%.0f", hit.distance * 3.28084)
        let key = "mesh_\(hit.label)_\(hit.direction)"

        // Same smart announcement logic — don't repeat unless distance changes or time passes
        if let prev = lastAnnounced[key] {
            let timeSince = now - prev.time
            let distChange = abs(hit.distance - prev.distance)
            if hit.distance >= 1.0 && timeSince < 15.0 { return }  // not close — wait 15s
            if hit.distance < 1.0 && timeSince < 4.0 { return }   // close — wait 4s
            if distChange < 0.5 && timeSince < 15.0 { return }    // hasn't moved much
        }

        switch hit.classification {
        case .wall:
            if hit.distance < RiskSolver.dangerThreshold {
                speech?.speak("Wall nearby\(dir)", urgency: 5.0)
                lastAnnounced[key] = (distance: hit.distance, time: now)
            } else if hit.distance < RiskSolver.cautionThreshold {
                speech?.speak("Wall\(dir), \(feet) feet", urgency: 4.0)
                lastAnnounced[key] = (distance: hit.distance, time: now)
            }
        case .door:
            speech?.speak("Door\(dir)", urgency: 3.5)
            lastAnnounced[key] = (distance: hit.distance, time: now)
            DispatchQueue.main.async { [weak self] in
                self?.onDetectionEvent?("door", hit.direction)
            }
        case .window:
            if hit.distance < 1.0 {
                speech?.speak("Glass ahead\(dir)", urgency: 5.0)
                lastAnnounced[key] = (distance: hit.distance, time: now)
            }
        case .seat:
            if hit.distance < RiskSolver.cautionThreshold {
                speech?.speak("Seat\(dir)", urgency: 3.5)
                lastAnnounced[key] = (distance: hit.distance, time: now)
            }
        case .table:
            if hit.distance < RiskSolver.cautionThreshold {
                speech?.speak("Table\(dir)", urgency: 3.5)
                lastAnnounced[key] = (distance: hit.distance, time: now)
            }
        default:
            break
        }

        #if DEBUG
        print("🏗 MESH  \(hit.label) \(hit.direction) \(String(format: "%.2f", hit.distance))m")
        #endif

        // Smart cloud AI trigger — fire when scene changes (different room/structure)
        let meshLabel = hit.label
        let meshNow = CACurrentMediaTime()
        if meshLabel != lastMeshLabel && meshNow - lastCloudScan > cloudMinInterval {
            lastMeshLabel = meshLabel
            lastCloudScan = meshNow
            if sceneDescriber.isReady, let frame = arSession?.currentFrame {
                print("☁️ Cloud AI triggered by scene change: \(meshLabel)")
                sceneDescriber.describeScene(pixelBuffer: frame.capturedImage)
            }
        }
    }

    // MARK: - Scene Segmentation (DeepLabV3)

    private func processSegmentation() {
        guard let result = sceneSegmenter.latestResult else { return }

        // Announce navigation-relevant objects with >15% frame coverage
        // that YOLO detection may have missed
        let yoloLabels = Set(latestDetections.compactMap { $0["label"] as? String })

        for (sceneClass, coverage) in result.detectedClasses {
            guard sceneClass.isNavigationRelevant, coverage > 0.15 else { continue }
            let label = sceneClass.label.lowercased()
            if yoloLabels.contains(label) { continue }   // YOLO already announced it

            let direction: String
            if result.centerClass == sceneClass         { direction = "ahead"    }
            else if result.leftClass == sceneClass      { direction = "on left"  }
            else                                         { direction = "on right" }

            speech?.speak("\(sceneClass.label) \(direction).", urgency: 2.0)  // detection level
            break  // one announcement per segmentation pass
        }
    }

    // MARK: - Public JS Bridge API

    /// Speak text via native SpeechController (called from JS `speak` message handler)
    /// Speak text from JS — high priority so it always interrupts LiDAR alerts
    func speakText(_ text: String) {
        speech?.speak(text, urgency: 7.0)  // user-initiated — highest priority
    }

    /// Trigger a manual AI scene scan (called from JS Scan/Describe button)
    func triggerManualScan() {
        speech?.speak("Scanning now.", urgency: 6.0)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
            self?.describeArea()
        }
    }

    // MARK: - Voice Command Handling

    func voiceCommandReceived(_ command: VoiceCommand) {
        handleVoiceCommand(command)
    }

    func handleVoiceCommand(_ command: VoiceCommand) {
        switch command {
        case .whatsAround:
            describeArea()
        case .isSafe:
            checkSafety()
        case .checkLeft:
            describeDirection("left", distance: sLeft, risk: RiskSolver.analyze(distance: sLeft))
        case .checkRight:
            describeDirection("right", distance: sRight, risk: RiskSolver.analyze(distance: sRight))
        case .stop:
            stop()
            speech?.speak("Paused. Say resume to continue.", urgency: 6.0)
        case .resume:
            start()
            speech?.speak("Resuming navigation.", urgency: 6.0)
        case .help:
            // High urgency so it doesn't get blocked by LiDAR alerts
            speech?.speak("Here are your controls. Double tap to scan. Hold the screen to speak. Swipe left or right to check sides. Voice commands: What's around. Is it safe. Left. Right. Scan. Read. What color. What bill. Remember this. Take me back. Stop. Resume. Help.", urgency: 10.0)
        case .scan:
            speech?.speak("Scanning now.", urgency: 6.0)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.describeArea()
            }
        case .readText:
            performReadText()
        case .identifyBill:
            performIdentifyBill()
        case .identifyColor:
            performIdentifyColor()
        case .rememberSpot:
            rememberCurrentSpot()
        case .guideBack:
            guideToSavedSpot()
        case .unknown(let text):
            speech?.speak("Sorry, I didn't catch that. Say help for commands.", urgency: 6.0)
        }
    }

    // MARK: - AR Breadcrumb ("remember this" / "take me back")

    /// Save the user's current pose as an ARAnchor so we can guide them back later.
    /// Replaces any previously saved spot.
    private func rememberCurrentSpot() {
        guard let session = arSession, let frame = session.currentFrame else {
            speech?.speak("Cannot remember spot. Camera not ready.", urgency: 7.0)
            return
        }
        if let old = savedAnchor { session.remove(anchor: old) }
        let anchor = ARAnchor(name: "savedSpot", transform: frame.camera.transform)
        session.add(anchor: anchor)
        savedAnchor = anchor
        speech?.speak("Spot saved. Say take me back to return here.", urgency: 7.0)
    }

    /// Speak the distance + bearing back to the saved spot, and start a periodic
    /// re-announcement timer so the user hears updates as they walk.
    private func guideToSavedSpot() {
        guard savedAnchor != nil else {
            speech?.speak("No saved spot. Say remember this first to mark your location.", urgency: 7.0)
            return
        }
        cancelGuideBackTimer()
        speech?.speak("Guiding you back.", urgency: 7.0)
        // First announcement after a short delay so the prompt above can finish.
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            self?.announceGuidanceStep(initial: true)
        }
        // Re-announce every 3.5s while we're still navigating.
        let t = Timer.scheduledTimer(withTimeInterval: 3.5, repeats: true) { [weak self] _ in
            self?.announceGuidanceStep(initial: false)
        }
        RunLoop.main.add(t, forMode: .common)
        guideBackTimer = t
    }

    private func cancelGuideBackTimer() {
        guideBackTimer?.invalidate()
        guideBackTimer = nil
        lastGuideAnnouncement = (0, 0)
    }

    /// Compute distance + bearing to the saved anchor and speak it. Stops the
    /// timer once we're within 0.5 m.
    private func announceGuidanceStep(initial: Bool) {
        guard let session = arSession,
              let anchor = savedAnchor,
              let frame = session.currentFrame else {
            cancelGuideBackTimer()
            speech?.speak("Lost saved spot. Say remember this again.", urgency: 7.0)
            return
        }

        let camTransform = frame.camera.transform
        let camPos = SIMD3<Float>(camTransform.columns.3.x,
                                   camTransform.columns.3.y,
                                   camTransform.columns.3.z)
        let anchorPos = SIMD3<Float>(anchor.transform.columns.3.x,
                                      anchor.transform.columns.3.y,
                                      anchor.transform.columns.3.z)
        let toAnchor = anchorPos - camPos

        // Project to horizontal plane (ignore vertical so steps/elevation don't skew the bearing).
        let dx = toAnchor.x, dz = toAnchor.z
        let horizDist = sqrt(dx * dx + dz * dz)

        // Arrived?
        if horizDist < 0.6 {
            speech?.speak("You arrived. Back at the saved spot.", urgency: 7.0)
            cancelGuideBackTimer()
            return
        }

        // Build the user's forward + right vectors in world space (horizontal).
        let camForward = -SIMD3<Float>(camTransform.columns.2.x,
                                        camTransform.columns.2.y,
                                        camTransform.columns.2.z)
        let camRight = SIMD3<Float>(camTransform.columns.0.x,
                                     camTransform.columns.0.y,
                                     camTransform.columns.0.z)

        let fwdH = simd_normalize(SIMD2<Float>(camForward.x, camForward.z))
        let rightH = simd_normalize(SIMD2<Float>(camRight.x, camRight.z))
        let toH = simd_normalize(SIMD2<Float>(dx, dz))

        // forwardDot = 1 → ahead; -1 → behind.
        // rightDot   = 1 → on right; -1 → on left.
        let forwardDot = simd_dot(fwdH, toH)
        let rightDot   = simd_dot(rightH, toH)

        // Signed angle from "where you face" to "where the anchor is".
        // Positive = anchor is to your right; negative = to your left.
        let angleDeg = atan2(rightDot, forwardDot) * 180 / .pi

        let direction: String
        let absA = abs(angleDeg)
        if absA < 20      { direction = "straight ahead" }
        else if absA < 60 { direction = angleDeg > 0 ? "slightly right" : "slightly left" }
        else if absA < 120 { direction = angleDeg > 0 ? "to your right" : "to your left" }
        else if absA < 160 { direction = angleDeg > 0 ? "behind you on the right" : "behind you on the left" }
        else              { direction = "directly behind you" }

        let feet = max(1, Int((horizDist * 3.28084).rounded()))

        // Suppress an update if nothing meaningful changed.
        let now = CACurrentMediaTime()
        let dd = abs(horizDist - lastGuideAnnouncement.distance)
        let dt = now - lastGuideAnnouncement.time
        if !initial && dd < 0.4 && dt < 6.0 { return }

        speech?.speak("\(feet) feet, \(direction).", urgency: 6.0)
        lastGuideAnnouncement = (horizDist, now)
    }

    // MARK: - Vision Tricks (user-initiated voice commands)

    private func performReadText() {
        guard let frame = arSession?.currentFrame else {
            speech?.speak("Camera not ready.", urgency: 7.0); return
        }
        // No prelude — back-to-back .user utterances trigger an
        // AVSpeechSynthesizer quirk where stopSpeaking-then-speak silently
        // drops the second utterance, swallowing the OCR result.
        VisionTricks.shared.recognizeText(in: frame.capturedImage) { [weak self] text in
            self?.speech?.speak(text, urgency: 7.0)
        }
    }

    private func performIdentifyBill() {
        guard let frame = arSession?.currentFrame else {
            speech?.speak("Camera not ready.", urgency: 7.0); return
        }
        VisionTricks.shared.identifyCurrency(in: frame.capturedImage) { [weak self] result in
            self?.speech?.speak(result, urgency: 7.0)
        }
    }

    private func performIdentifyColor() {
        guard let frame = arSession?.currentFrame else {
            speech?.speak("Camera not ready.", urgency: 7.0); return
        }
        let color = VisionTricks.shared.identifyColor(in: frame.capturedImage)
        speech?.speak("That looks \(color).", urgency: 7.0)
    }

    private func describeArea() {
        isScanActive = true  // suppress distance band speech
        describeAreaLocal()

        isUserScan = true
        if sceneDescriber.isReady, let frame = arSession?.currentFrame {
            sceneDescriber.describeScene(pixelBuffer: frame.capturedImage)
        } else {
            isUserScan = false
        }

        // Re-enable distance bands after 5s (enough for scan to finish)
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) { [weak self] in
            self?.isScanActive = false
        }
    }

    /// Local-only scene description (no API calls) — used as fallback
    private func describeAreaLocal() {
        var parts: [String] = []

        if let d = sCenter {
            let feet = d * 3.28084
            let risk = RiskSolver.analyze(distance: d)
            parts.append("Ahead, \(String(format: "%.0f", feet)) feet, \(riskString(risk))")
        } else {
            parts.append("Ahead is clear")
        }
        if let d = sLeft {
            parts.append("Left, \(String(format: "%.0f", d * 3.28084)) feet")
        }
        if let d = sRight {
            parts.append("Right, \(String(format: "%.0f", d * 3.28084)) feet")
        }

        // Include latest detections
        for det in latestDetections.prefix(2) {
            if let label = det["label"] as? String, let dir = det["direction"] as? String {
                parts.append("\(label) \(dir)")
            }
        }

        speech?.speak(parts.joined(separator: ". "), urgency: 7.0)  // user asked for this
    }

    private func checkSafety() {
        let riskC = RiskSolver.analyze(distance: sCenter)
        let riskL = RiskSolver.analyze(distance: sLeft)
        let riskR = RiskSolver.analyze(distance: sRight)

        if riskC == .safe && riskL == .safe && riskR == .safe {
            speech?.speak("Path is clear. Safe to proceed.", urgency: 6.0)
        } else if riskC == .danger {
            speech?.speak("Careful, something directly ahead.", urgency: 6.0)
        } else if riskC == .caution {
            speech?.speak("Proceed with caution. Something ahead.", urgency: 6.0)
        } else {
            var warning = "Center is clear."
            if riskL != .safe { warning += " Obstacle on left." }
            if riskR != .safe { warning += " Obstacle on right." }
            speech?.speak(warning, urgency: 6.0)
        }
    }

    private func describeDirection(_ direction: String, distance: Float?, risk: RiskLevel) {
        if let d = distance {
            let feet = d * 3.28084
            speech?.speak("\(direction.capitalized), \(String(format: "%.0f", feet)) feet, \(riskString(risk)).", urgency: 6.0)
        } else {
            speech?.speak("\(direction.capitalized) side is clear.", urgency: 6.0)
        }
    }

    // MARK: - Helpers

    private func riskString(_ risk: RiskLevel) -> String {
        switch risk {
        case .safe: return "safe"
        case .caution: return "caution"
        case .danger: return "danger"
        }
    }

    private func getPercentile(_ arr: [Float]) -> Float? {
        if arr.isEmpty { return nil }
        let sorted = arr.sorted()
        let i = min(Int(Float(sorted.count) * 0.2), sorted.count - 1)
        return sorted[i]
    }

    private func getMedian(_ arr: [Float]) -> Float? {
        if arr.isEmpty { return nil }
        let sorted = arr.sorted()
        return sorted[sorted.count / 2]
    }

    private func smooth(_ old: Float?, _ new: Float?, _ a: Float) -> Float? {
        guard let n = new else { return old }
        guard let o = old else { return n }
        return o + a * (n - o)
    }
}
