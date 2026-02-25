import UIKit
import CoreML

// MARK: - Stage 2 Processor
// Handles: Rectified image → Model inference → Signal extraction → ECG leads

class Stage2Processor {
    
    private let model: MLModel
    
    init(modelURL: URL) throws {
        let config = MLModelConfiguration()
        #if targetEnvironment(simulator)
        config.computeUnits = .cpuOnly
        #else
        config.computeUnits = .all
        #endif
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
    }
    
    // MARK: - Main Processing
    
    struct Stage2Result {
        let rowSignals: [[Float]]
        let leads: [String: [Float]]
        let rhythmLead: [Float]
        let signalLength: Int
    }
    
    /// Process rectified image through Stage 2
    func process(rectifiedImage: UIImage, targetLength: Int = 5000) throws -> Stage2Result {
        print("[Stage2] Starting processing...")
        print("[Stage2] Input size: \(rectifiedImage.size)")
        
        // 1. Crop to Stage 2 input size
        let cropRect = CGRect(
            x: 0, y: 0,
            width: ECGConstants.stage2Width,
            height: ECGConstants.stage2Height
        )
        
        let croppedImage: UIImage
        if let cropped = ImageUtils.crop(image: rectifiedImage, to: cropRect) {
            croppedImage = cropped
        } else {
            // If crop fails, resize instead
            croppedImage = ImageUtils.resize(
                image: rectifiedImage,
                to: CGSize(width: ECGConstants.stage2Width, height: ECGConstants.stage2Height)
            ) ?? rectifiedImage
        }
        
        print("[Stage2] Cropped size: \(croppedImage.size)")
        
        // 2. Convert to pixel buffer
        guard let pixelBuffer = ImageUtils.pixelBuffer(
            from: croppedImage,
            width: ECGConstants.stage2Width,
            height: ECGConstants.stage2Height
        ) else {
            throw ECGError.pixelBufferCreationFailed
        }
        
        // 3. Run model inference
        print("[Stage2] Running model inference...")
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
        let output = try model.prediction(from: input)
        
        // 4. Extract pixel output
        guard let pixelArray = output.featureValue(for: "pixel")?.multiArrayValue else {
            throw ECGError.modelOutputMissing("pixel output missing")
        }
        
        let pixelData = pixelArray.toFloatArray()
        let shape = pixelArray.shapeArray
        
        print("[Stage2] pixel shape: \(shape)")
        
        // Shape is [1, 4, 1696, 2176] or [4, 1696, 2176]
        let numRows = shape.count == 4 ? shape[1] : shape[0]
        let height = shape.count == 4 ? shape[2] : shape[1]
        let width = shape.count == 4 ? shape[3] : shape[2]
        
        // 5. Extract signals
        let (t0, t1) = ECGConstants.timespan
        let rowSignals = extractSignals(
            pixelData: pixelData,
            numRows: numRows,
            height: height,
            width: width,
            timeStart: t0,
            timeEnd: t1,
            targetLength: targetLength
        )
        
        // Print signal stats
        for (i, signal) in rowSignals.enumerated() {
            let minVal = signal.min() ?? 0
            let maxVal = signal.max() ?? 0
            print("[Stage2] Row \(i): min=\(minVal), max=\(maxVal)")
        }
        
        // 6. Split into leads
        let (leads, rhythmLead) = splitIntoLeads(
            rowSignals: rowSignals,
            targetLength: targetLength
        )
        
        print("[Stage2] Extracted \(leads.count) leads")
        
        return Stage2Result(
            rowSignals: rowSignals,
            leads: leads,
            rhythmLead: rhythmLead,
            signalLength: targetLength
        )
    }
    
    // MARK: - Signal Extraction
    
    private func extractSignals(
        pixelData: [Float],
        numRows: Int,
        height: Int,
        width: Int,
        timeStart: Int,
        timeEnd: Int,
        targetLength: Int
    ) -> [[Float]] {
        
        let cropWidth = timeEnd - timeStart
        var rowSignals = [[Float]]()
        
        for row in 0..<numRows {
            var yPositions = [Float](repeating: 0, count: cropWidth)
            let zeroMV = ECGConstants.zeroMV[row]
            
            for x in 0..<cropWidth {
                let xIdx = x + timeStart
                
                // Find argmax y-position
                var maxVal: Float = -1
                var maxY = Int(zeroMV)
                
                for y in 0..<height {
                    let idx = row * height * width + y * width + xIdx
                    if idx < pixelData.count {
                        let val = pixelData[idx]
                        if val > maxVal {
                            maxVal = val
                            maxY = y
                        }
                    }
                }
                
                // If no strong detection, use baseline
                if maxVal < ECGConstants.signalThreshold {
                    maxY = Int(zeroMV)
                }
                
                yPositions[x] = Float(maxY)
            }
            
            // Convert to mV
            var signalMV = yPositions.map { (zeroMV - $0) / ECGConstants.mvToPixel }
            
            // Clip to reasonable range
            signalMV = signalMV.map { min(max($0, -5), 5) }
            
            // Interpolate if needed
            if cropWidth != targetLength {
                signalMV = interpolateSignal(signalMV, toLength: targetLength)
            }
            
            rowSignals.append(signalMV)
        }
        
        return rowSignals
    }
    
    private func interpolateSignal(_ signal: [Float], toLength targetLength: Int) -> [Float] {
        let srcLength = signal.count
        guard srcLength > 1 else { return Array(repeating: 0, count: targetLength) }
        
        var result = [Float](repeating: 0, count: targetLength)
        
        for i in 0..<targetLength {
            let srcIdx = Float(i) / Float(targetLength - 1) * Float(srcLength - 1)
            let idx0 = Int(floor(srcIdx))
            let idx1 = min(idx0 + 1, srcLength - 1)
            let frac = srcIdx - Float(idx0)
            
            result[i] = signal[idx0] * (1 - frac) + signal[idx1] * frac
        }
        
        return result
    }
    
    // MARK: - Lead Splitting
    
    private func splitIntoLeads(
        rowSignals: [[Float]],
        targetLength: Int
    ) -> ([String: [Float]], [Float]) {
        
        var leads = [String: [Float]]()
        let segmentLength = targetLength / 4
        
        // Rows 0-2: each has 4 leads
        for rowIdx in 0..<min(3, rowSignals.count) {
            let rowSignal = rowSignals[rowIdx]
            let leadNames = ECGConstants.leadNames[rowIdx]
            
            for (segIdx, leadName) in leadNames.enumerated() {
                let startIdx = segIdx * segmentLength
                let endIdx = min(startIdx + segmentLength, rowSignal.count)
                
                if startIdx < rowSignal.count {
                    leads[leadName] = Array(rowSignal[startIdx..<endIdx])
                }
            }
        }
        
        // Row 3: full II rhythm
        let rhythmLead = rowSignals.count > 3 ? rowSignals[3] : []
        
        return (leads, rhythmLead)
    }
}
