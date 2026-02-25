import UIKit
import CoreML

// MARK: - ECG Pipeline
// Main orchestrator that runs all 3 stages sequentially

class ECGPipeline {
    
    private let stage0: Stage0Processor
    private let stage1: Stage1Processor
    private let stage2: Stage2Processor
    
    // MARK: - Initialization
    
    init(stage0URL: URL, stage1URL: URL, stage2URL: URL) throws {
        print("[Pipeline] Initializing with model URLs...")
        self.stage0 = try Stage0Processor(modelURL: stage0URL)
        self.stage1 = try Stage1Processor(modelURL: stage1URL)
        self.stage2 = try Stage2Processor(modelURL: stage2URL)
        print("[Pipeline] All models loaded successfully")
    }
    
    convenience init() throws {
        // Try compiled models first (.mlmodelc), then packages (.mlpackage)
        guard let stage0URL = Bundle.main.url(forResource: "Stage0", withExtension: "mlmodelc")
                ?? Bundle.main.url(forResource: "Stage0", withExtension: "mlpackage"),
              let stage1URL = Bundle.main.url(forResource: "Stage1", withExtension: "mlmodelc")
                ?? Bundle.main.url(forResource: "Stage1", withExtension: "mlpackage"),
              let stage2URL = Bundle.main.url(forResource: "Stage2", withExtension: "mlmodelc")
                ?? Bundle.main.url(forResource: "Stage2", withExtension: "mlpackage") else {
            throw ECGPipelineError.modelsNotFound
        }
        
        // Compile .mlpackage if needed
        let s0 = stage0URL.pathExtension == "mlpackage" ? try MLModel.compileModel(at: stage0URL) : stage0URL
        let s1 = stage1URL.pathExtension == "mlpackage" ? try MLModel.compileModel(at: stage1URL) : stage1URL
        let s2 = stage2URL.pathExtension == "mlpackage" ? try MLModel.compileModel(at: stage2URL) : stage2URL
        
        try self.init(stage0URL: s0, stage1URL: s1, stage2URL: s2)
    }
    
    // MARK: - Main Processing
    
    func process(
        image: UIImage,
        targetSignalLength: Int = 5000,
        progressHandler: ((Float, String) -> Void)? = nil
    ) -> ECGPipelineResult {
        
        let startTime = Date()
        
        do {
            // Stage 0
            progressHandler?(0.1, "Step 1: Detecting orientation...")
            let stage0Result = try stage0.process(image: image)
            
            let stage0Info = stage0Result.usedFallback ? " (fallback)" : ""
            progressHandler?(0.3, "Step 1 complete\(stage0Info)")
            
            // Stage 1
            progressHandler?(0.4, "Step 2: Detecting grid...")
            let stage1Result = try stage1.process(normalizedImage: stage0Result.normalizedImage)
            
            let stage1Info = stage1Result.usedFallback ? " (fallback)" : ""
            progressHandler?(0.6, "Step 2 complete\(stage1Info)")
            
            // Stage 2
            progressHandler?(0.7, "Step 3: Extracting signals...")
            let stage2Result = try stage2.process(
                rectifiedImage: stage1Result.rectifiedImage,
                targetLength: targetSignalLength
            )
            
            progressHandler?(0.9, "Step 3 complete")
            
            // Build result
            let processingTime = Date().timeIntervalSince(startTime)
            
            var signals = [String: ECGSignal]()
            
            // Add 12 leads
            for (leadName, samples) in stage2Result.leads {
                signals[leadName] = ECGSignal(
                    leadName: leadName,
                    samples: samples,
                    sampleRate: Float(targetSignalLength) / 2.5
                )
            }
            
            // Add II rhythm
            if !stage2Result.rhythmLead.isEmpty {
                signals["II-rhythm"] = ECGSignal(
                    leadName: "II-rhythm",
                    samples: stage2Result.rhythmLead,
                    sampleRate: Float(targetSignalLength) / 10.0
                )
            }
            
            progressHandler?(1.0, String(format: "Done in %.1fs", processingTime))
            
            return ECGPipelineResult(
                success: true,
                signals: signals,
                normalizedImage: stage0Result.normalizedImage.cgImage,
                rectifiedImage: stage1Result.rectifiedImage.cgImage,
                errorMessage: nil,
                keypoints: stage0Result.keypoints,
                homography: stage0Result.homography,
                stage0UsedFallback: stage0Result.usedFallback,
                stage1UsedFallback: stage1Result.usedFallback,
                rowSignals: stage2Result.rowSignals
            )
            
        } catch {
            print("[Pipeline] Error: \(error)")
            return ECGPipelineResult.failure("Pipeline failed: \(error.localizedDescription)")
        }
    }
    
    /// Process asynchronously
    func processAsync(
        image: UIImage,
        targetSignalLength: Int = 5000,
        progressHandler: ((Float, String) -> Void)? = nil,
        completion: @escaping (ECGPipelineResult) -> Void
    ) {
        DispatchQueue.global(qos: .userInitiated).async {
            let result = self.process(
                image: image,
                targetSignalLength: targetSignalLength,
                progressHandler: { progress, message in
                    DispatchQueue.main.async {
                        progressHandler?(progress, message)
                    }
                }
            )
            
            DispatchQueue.main.async {
                completion(result)
            }
        }
    }
}

// MARK: - Pipeline Errors

enum ECGPipelineError: Error, LocalizedError {
    case modelsNotFound
    case invalidImage
    case processingFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .modelsNotFound:
            return "CoreML models not found in bundle. Make sure Stage0, Stage1, Stage2 models are included."
        case .invalidImage:
            return "Invalid input image"
        case .processingFailed(let reason):
            return "Processing failed: \(reason)"
        }
    }
}
