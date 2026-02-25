import Foundation
import CoreML
import Accelerate

// MARK: - Classification Result

struct ClassificationResult {
    let predictions: [(className: String, fullName: String, confidence: Float, isPositive: Bool)]
    let rawProbabilities: [String: Float]
    let positiveCount: Int
    
    var primaryDiagnosis: String {
        // Return highest confidence positive prediction, or "Normal" if NORM is positive
        let positives = predictions.filter { $0.isPositive }
        if positives.isEmpty {
            return "No significant findings"
        }
        if let norm = positives.first(where: { $0.className == "NORM" }) {
            return norm.fullName
        }
        return positives.first?.fullName ?? "Unknown"
    }
    
    var primaryConfidence: Float {
        let positives = predictions.filter { $0.isPositive }
        if positives.isEmpty {
            return 0
        }
        if let norm = positives.first(where: { $0.className == "NORM" }) {
            return norm.confidence
        }
        return positives.first?.confidence ?? 0
    }
}

// MARK: - ECG Classifier Processor

class ECGClassifierProcessor {
    
    private var model: MLModel?
    private let labelNames = ClassifierConstants.labelNames
    private let thresholds = ClassifierConstants.thresholds
    private let signalMean = ClassifierConstants.signalMean
    private let signalStd = ClassifierConstants.signalStd
    
    enum ClassifierError: Error {
        case modelNotLoaded
        case invalidInput
        case preprocessingFailed
        case inferenceFailed
    }
    
    // MARK: - Initialization
    
    init(modelURL: URL? = nil) throws {
        if let url = modelURL {
            try loadModel(from: url)
        } else {
            // Try to find model in bundle
            if let bundleURL = Bundle.main.url(forResource: "ECGClassifier", withExtension: "mlmodelc") {
                try loadModel(from: bundleURL)
            }
        }
    }
    
    func loadModel(from url: URL) throws {
        let config = MLModelConfiguration()
        #if targetEnvironment(simulator)
        config.computeUnits = .cpuOnly
        #else
        config.computeUnits = .all
        #endif
        model = try MLModel(contentsOf: url, configuration: config)
        print("[Classifier] Model loaded successfully")
    }
    
    var isModelLoaded: Bool {
        return model != nil
    }
    
    // MARK: - Classification
    
    /// Classify ECG signals
    /// - Parameter signals: Dictionary of lead name to signal samples (in mV)
    /// - Returns: Classification result with predictions
    func classify(signals: [String: ECGSignal]) throws -> ClassificationResult {
        guard model != nil else {
            throw ClassifierError.modelNotLoaded
        }
        
        // Assemble signal array [5000, 12]
        let assembledSignal = try assembleSignal(from: signals)
        
        // Preprocess (normalize)
        let preprocessed = preprocess(assembledSignal)
        
        // Run inference
        let logits = try runInference(preprocessed)
        
        // Apply sigmoid to get probabilities
        let probabilities = sigmoid(logits)
        
        // Build results
        return buildResult(probabilities: probabilities)
    }
    
    /// Classify from raw signal array
    /// - Parameter signal: Array of shape [5000, 12] in mV
    func classify(signal: [[Float]]) throws -> ClassificationResult {
        guard model != nil else {
            throw ClassifierError.modelNotLoaded
        }
        
        guard signal.count == ClassifierConstants.expectedSamples,
              signal.first?.count == ClassifierConstants.expectedLeads else {
            throw ClassifierError.invalidInput
        }
        
        // Flatten to [5000 * 12]
        let flattened = signal.flatMap { $0 }
        
        // Preprocess
        let preprocessed = preprocess(flattened)
        
        // Run inference
        let logits = try runInference(preprocessed)
        
        // Apply sigmoid
        let probabilities = sigmoid(logits)
        
        return buildResult(probabilities: probabilities)
    }
    
    // MARK: - Signal Assembly
    
    private func assembleSignal(from signals: [String: ECGSignal]) throws -> [Float] {
        // Create [5000, 12] array
        var assembled = [Float](repeating: 0, count: ClassifierConstants.expectedSamples * ClassifierConstants.expectedLeads)
        
        // Lead order
        let leadOrder = ClassifierConstants.leadNames
        
        // Time windows for each lead (in samples)
        let leadWindows: [String: (start: Int, end: Int)] = [
            "I": (0, 1250),
            "II": (0, 5000),  // II rhythm strip is full length
            "III": (0, 1250),
            "aVR": (1250, 2500),
            "aVL": (1250, 2500),
            "aVF": (1250, 2500),
            "V1": (2500, 3750),
            "V2": (2500, 3750),
            "V3": (2500, 3750),
            "V4": (3750, 5000),
            "V5": (3750, 5000),
            "V6": (3750, 5000)
        ]
        
        for (leadIdx, leadName) in leadOrder.enumerated() {
            guard let signal = signals[leadName] else {
                print("[Classifier] Warning: Missing lead \(leadName)")
                continue
            }
            
            guard let window = leadWindows[leadName] else { continue }
            
            let samples = signal.samples
            let windowLength = window.end - window.start
            
            // Copy samples to appropriate time window
            for i in 0..<min(samples.count, windowLength) {
                let sampleIdx = window.start + i
                if sampleIdx < ClassifierConstants.expectedSamples {
                    assembled[sampleIdx * ClassifierConstants.expectedLeads + leadIdx] = samples[i]
                }
            }
        }
        
        return assembled
    }
    
    // MARK: - Preprocessing
    
    private func preprocess(_ signal: [Float]) -> [Float] {
        // Normalize using per-lead mean and std
        var normalized = [Float](repeating: 0, count: signal.count)
        
        let numSamples = ClassifierConstants.expectedSamples
        let numLeads = ClassifierConstants.expectedLeads
        
        for sampleIdx in 0..<numSamples {
            for leadIdx in 0..<numLeads {
                let idx = sampleIdx * numLeads + leadIdx
                let value = signal[idx]
                
                // Replace NaN/Inf with 0
                if value.isNaN || value.isInfinite {
                    normalized[idx] = 0
                } else {
                    // Z-score normalization
                    normalized[idx] = (value - signalMean[leadIdx]) / (signalStd[leadIdx] + 1e-8)
                }
            }
        }
        
        return normalized
    }
    
    // MARK: - Inference
    
    private func runInference(_ signal: [Float]) throws -> [Float] {
        guard let model = model else {
            throw ClassifierError.modelNotLoaded
        }
        
        // Create MLMultiArray [1, 5000, 12]
        let shape: [NSNumber] = [1, NSNumber(value: ClassifierConstants.expectedSamples), NSNumber(value: ClassifierConstants.expectedLeads)]
        guard let inputArray = try? MLMultiArray(shape: shape, dataType: .float32) else {
            throw ClassifierError.preprocessingFailed
        }
        
        // Copy data
        let ptr = inputArray.dataPointer.bindMemory(to: Float.self, capacity: signal.count)
        for i in 0..<signal.count {
            ptr[i] = signal[i]
        }
        
        // Create input
        let inputName = "signal"  // Must match CoreML model input name
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: inputArray)])
        
        // Run inference
        guard let output = try? model.prediction(from: input) else {
            throw ClassifierError.inferenceFailed
        }
        
        // Extract logits
        guard let logitsArray = output.featureValue(for: "logits")?.multiArrayValue else {
            throw ClassifierError.inferenceFailed
        }
        
        // Convert to [Float]
        var logits = [Float](repeating: 0, count: labelNames.count)
        let logitsPtr = logitsArray.dataPointer.bindMemory(to: Float.self, capacity: labelNames.count)
        for i in 0..<labelNames.count {
            logits[i] = logitsPtr[i]
        }
        
        print("[Classifier] Logits: \(logits)")
        
        return logits
    }
    
    // MARK: - Post-processing
    
    private func sigmoid(_ x: [Float]) -> [Float] {
        return x.map { 1.0 / (1.0 + exp(-$0)) }
    }
    
    private func buildResult(probabilities: [Float]) -> ClassificationResult {
        var predictions: [(className: String, fullName: String, confidence: Float, isPositive: Bool)] = []
        var rawProbabilities: [String: Float] = [:]
        var positiveCount = 0
        
        for (i, prob) in probabilities.enumerated() {
            let className = labelNames[i]
            let fullName = ClassifierConstants.classFullNames[className] ?? className
            let threshold = thresholds[i]
            let isPositive = prob > threshold
            
            predictions.append((className, fullName, prob, isPositive))
            rawProbabilities[className] = prob
            
            if isPositive {
                positiveCount += 1
            }
        }
        
        // Sort by confidence (descending)
        predictions.sort { $0.confidence > $1.confidence }
        
        print("[Classifier] Probabilities: \(rawProbabilities)")
        print("[Classifier] Positive predictions: \(positiveCount)")
        
        return ClassificationResult(
            predictions: predictions,
            rawProbabilities: rawProbabilities,
            positiveCount: positiveCount
        )
    }
}
