import Foundation
import CoreGraphics

// MARK: - ECG Pipeline Constants

struct ECGConstants {
    
    // MARK: - Image Dimensions
    
    /// Stage 0/1 normalized image size
    static let normalizedWidth: Int = 1440
    static let normalizedHeight: Int = 1152
    
    /// Rectified image size (output of Stage 1)
    static let rectifiedWidth: Int = 2200
    static let rectifiedHeight: Int = 1700
    
    /// Stage 2 input size (cropped from rectified)
    static let stage2Width: Int = 2176
    static let stage2Height: Int = 1696
    
    // MARK: - Signal Extraction Parameters
    
    /// Y-position of 0mV baseline for each of 4 rows
    static let zeroMV: [Float] = [703.5, 987.5, 1271.5, 1531.5]
    
    /// Pixels per millivolt
    static let mvToPixel: Float = 79.0
    
    /// X-range for signal extraction (timespan)
    static let timespan: (start: Int, end: Int) = (118, 2080)
    
    // MARK: - Lead Names
    
    /// Lead labels for each row (rows 0-2 have 4 leads each, row 3 is full II)
    static let leadNames: [[String]] = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"]
    ]
    
    /// All 12 lead names in standard order
    static let allLeadNames: [String] = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]
    
    // MARK: - Label Mapping
    
    /// Label to lead name mapping (from Stage 0 marker output)
    static let labelToLeadName: [Int: String] = [
        0: "None", 1: "I", 2: "aVR", 3: "V1", 4: "V4",
        5: "II", 6: "aVL", 7: "V2", 8: "V5",
        9: "III", 10: "aVF", 11: "V3", 12: "V6", 13: "II-rhythm"
    ]
    
    /// Labels used for keypoint extraction (9 keypoints)
    /// Order: aVR, V1, V4, aVL, V2, V5, aVF, V3, V6
    static let keypointLabels: [Int] = [2, 3, 4, 6, 7, 8, 10, 11, 12]
    
    // MARK: - Reference Points for Homography
    
    /// Reference 9 keypoints for homography (from Python REF_PT9)
    /// Order matches keypointLabels: aVR, V1, V4, aVL, V2, V5, aVF, V3, V6
    static let referencePoints9: [(x: Float, y: Float)] = [
        (x: 440.5, y: 529.3),   // aVR (label 2)
        (x: 715.2, y: 529.3),   // V1  (label 3)
        (x: 1013.1, y: 529.3),  // V4  (label 4)
        (x: 440.5, y: 689.9),   // aVL (label 6)
        (x: 715.2, y: 689.9),   // V2  (label 7)
        (x: 1013.1, y: 689.9),  // V5  (label 8)
        (x: 440.5, y: 849.9),   // aVF (label 10)
        (x: 715.2, y: 849.9),   // V3  (label 11)
        (x: 1013.1, y: 849.9)   // V6  (label 12)
    ]
    
    // MARK: - Grid Dimensions
    
    static let gridRows: Int = 44
    static let gridCols: Int = 57
    
    // MARK: - Thresholds
    
    /// Minimum area for connected component (keypoints)
    static let minComponentArea: Int = 10
    
    /// Threshold for gridpoint detection
    static let gridpointThreshold: Float = 0.5
    
    /// Threshold for signal detection
    static let signalThreshold: Float = 0.05
}

// MARK: - Keypoint Structure

struct ECGKeypoint {
    var x: Float
    var y: Float
    let label: Int
    let leadName: String
    var isMatched: Bool = false
    
    init(x: Float, y: Float, label: Int) {
        self.x = x
        self.y = y
        self.label = label
        self.leadName = ECGConstants.labelToLeadName[label] ?? "Unknown"
    }
}

// MARK: - ECG Signal Structure

struct ECGSignal {
    let leadName: String
    let samples: [Float]
    let sampleRate: Float
    
    var duration: Float { Float(samples.count) / sampleRate }
    var minValue: Float { samples.min() ?? 0 }
    var maxValue: Float { samples.max() ?? 0 }
}

// MARK: - Pipeline Result

struct ECGPipelineResult {
    let success: Bool
    let signals: [String: ECGSignal]
    let normalizedImage: CGImage?
    let rectifiedImage: CGImage?
    let errorMessage: String?
    
    // Debug info
    let keypoints: [ECGKeypoint]?
    let homography: [[Double]]?
    let stage0UsedFallback: Bool
    let stage1UsedFallback: Bool
    let rowSignals: [[Float]]?  // Raw row signals for overlay
    
    static func failure(_ message: String) -> ECGPipelineResult {
        ECGPipelineResult(
            success: false, signals: [:], normalizedImage: nil,
            rectifiedImage: nil, errorMessage: message,
            keypoints: nil, homography: nil,
            stage0UsedFallback: false, stage1UsedFallback: false,
            rowSignals: nil
        )
    }
}
