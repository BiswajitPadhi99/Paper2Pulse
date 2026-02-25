import UIKit
import CoreML

// MARK: - Stage 0 Processor
// Handles: Image preprocessing → Model inference → Keypoint extraction → Homography → Normalized image

class Stage0Processor {
    
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
    
    // MARK: - Main Processingre
    
    struct Stage0Result {
        let normalizedImage: UIImage
        let keypoints: [ECGKeypoint]
        let orientation: Int
        let homography: [[Double]]?
        let usedFallback: Bool
    }
    
    /// Process image through Stage 0
    func process(image: UIImage) throws -> Stage0Result {
        print("[Stage0] Starting processing...")
        print("[Stage0] Input image size: \(image.size)")
        
        // 1. Preprocess: resize and pad
        guard let preprocessed = ImageUtils.resizeAndPad(
            image: image,
            targetWidth: ECGConstants.normalizedWidth,
            padToMultiple: 32
        ) else {
            throw ECGError.preprocessingFailed("Failed to resize/pad image")
        }
        
        let paddedImage = preprocessed.image
        let scale = preprocessed.scale
        let scaledHeight = preprocessed.scaledHeight
        let paddedWidth = Int(preprocessed.paddedSize.width)
        let paddedHeight = Int(preprocessed.paddedSize.height)
        
        print("[Stage0] Padded size: \(paddedWidth)x\(paddedHeight), scale: \(scale)")
        
        // 2. Convert to pixel buffer
        guard let pixelBuffer = ImageUtils.pixelBuffer(
            from: paddedImage,
            width: paddedWidth,
            height: paddedHeight
        ) else {
            throw ECGError.pixelBufferCreationFailed
        }
        
        // 3. Run model inference
        print("[Stage0] Running model inference...")
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
        let output = try model.prediction(from: input)
        
        // 4. Extract marker and orientation outputs
        guard let markerArray = output.featureValue(for: "marker")?.multiArrayValue,
              let orientationArray = output.featureValue(for: "orientation")?.multiArrayValue else {
            throw ECGError.modelOutputMissing("marker or orientation output missing")
        }
        
        let markerData = markerArray.toFloatArray()
        let orientationData = orientationArray.toFloatArray()
        
        print("[Stage0] Marker shape: \(markerArray.shapeArray)")
        print("[Stage0] Orientation shape: \(orientationArray.shapeArray)")
        
        // 5. Get orientation (argmax)
        let orientationClass = getOrientation(orientationData)
        let rotationCount = orientationToRotation(orientationClass)
        print("[Stage0] Detected orientation: \(orientationClass), rotation: \(rotationCount)")
        
        // 6. Rotate image if needed
        let rotatedImage = ImageUtils.rotate(image: image, by: rotationCount)
        
        // 7. Extract keypoints from marker
        let markerShape = markerArray.shapeArray
        let channels = markerShape.count == 4 ? markerShape[1] : markerShape[0]
        let markerHeight = markerShape.count == 4 ? markerShape[2] : markerShape[1]
        let markerWidth = markerShape.count == 4 ? markerShape[3] : markerShape[2]
        
        let allKeypoints = extractKeypoints(
            markerData: markerData,
            channels: channels,
            height: min(markerHeight, scaledHeight),
            width: markerWidth,
            dataHeight: markerHeight,  // Actual tensor height for stride calculation
            dataWidth: markerWidth,    // Actual tensor width for stride calculation
            scale: scale,
            rotationCount: rotationCount
        )
        
        // Filter out invalid keypoints (0, 0)
        let validKeypoints = allKeypoints.filter { $0.x > 0 && $0.y > 0 }
        print("[Stage0] Found \(validKeypoints.count)/\(allKeypoints.count) valid keypoints")
        
        // 8. Try to compute homography, fallback to simple resize if fails
        var homography: [[Double]]? = nil
        var normalizedImage: UIImage
        var usedFallback = false
        
        if validKeypoints.count >= 4 {
            // Get matching reference points for valid keypoints
            let validIndices = allKeypoints.enumerated().compactMap { index, kp -> Int? in
                (kp.x > 0 && kp.y > 0) ? index : nil
            }
            
            let srcPoints = validKeypoints.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
            let dstPoints = validIndices.map { CGPoint(x: CGFloat(ECGConstants.referencePoints9[$0].x),
                                                        y: CGFloat(ECGConstants.referencePoints9[$0].y)) }
            
            print("[Stage0] Computing homography with \(srcPoints.count) points...")
            
            if let (H, _) = HomographyComputer.findHomography(
                srcPoints: srcPoints,
                dstPoints: dstPoints,
                useRANSAC: true,
                threshold: 10.0
            ) {
                homography = H
                print("[Stage0] Homography computed successfully")
                
                // Apply homography
                if let warped = ImageUtils.warpPerspective(
                    image: rotatedImage,
                    homography: H,
                    outputSize: CGSize(
                        width: ECGConstants.normalizedWidth,
                        height: ECGConstants.normalizedHeight
                    )
                ) {
                    normalizedImage = warped
                    print("[Stage0] Perspective warp applied")
                } else {
                    print("[Stage0] Warp failed, using fallback")
                    normalizedImage = fallbackNormalize(image: rotatedImage)
                    usedFallback = true
                }
            } else {
                print("[Stage0] Homography computation failed, using fallback")
                normalizedImage = fallbackNormalize(image: rotatedImage)
                usedFallback = true
            }
        } else {
            print("[Stage0] Not enough keypoints (\(validKeypoints.count)), using fallback")
            normalizedImage = fallbackNormalize(image: rotatedImage)
            usedFallback = true
        }
        
        return Stage0Result(
            normalizedImage: normalizedImage,
            keypoints: allKeypoints,
            orientation: rotationCount,
            homography: homography,
            usedFallback: usedFallback
        )
    }
    
    /// Fallback: simple resize without perspective correction
    private func fallbackNormalize(image: UIImage) -> UIImage {
        let targetSize = CGSize(width: ECGConstants.normalizedWidth,
                                height: ECGConstants.normalizedHeight)
        return ImageUtils.resize(image: image, to: targetSize) ?? image
    }
    
    // MARK: - Orientation Processing
    
    private func getOrientation(_ data: [Float]) -> Int {
        let numClasses = 8
        var avgProbs = [Float](repeating: 0, count: numClasses)
        
        let batchSize = data.count / numClasses
        for b in 0..<batchSize {
            for c in 0..<numClasses {
                avgProbs[c] += data[b * numClasses + c]
            }
        }
        
        var maxVal: Float = -Float.infinity
        var maxIdx = 0
        for i in 0..<numClasses {
            if avgProbs[i] > maxVal {
                maxVal = avgProbs[i]
                maxIdx = i
            }
        }
        
        return maxIdx
    }
    
    private func orientationToRotation(_ orientation: Int) -> Int {
        switch orientation {
        case 0, 4: return 0
        case 1, 5: return -1
        case 2, 6: return -2
        case 3, 7: return -3
        default: return 0
        }
    }
    
    // MARK: - Keypoint Extraction
    
    private func extractKeypoints(
        markerData: [Float],
        channels: Int,
        height: Int,
        width: Int,
        dataHeight: Int,
        dataWidth: Int,
        scale: Float,
        rotationCount: Int
    ) -> [ECGKeypoint] {
        
        print("[Stage0] Extracting keypoints from marker data: channels=\(channels), height=\(height), width=\(width), dataHeight=\(dataHeight)")
        
        let classMap = ConnectedComponents.argmaxAlongChannels(
            data: markerData,
            channels: channels,
            height: height,
            width: width,
            dataHeight: dataHeight,
            dataWidth: dataWidth
        )
        
        // Debug: count how many pixels of each label
        var labelCounts = [Int: Int]()
        for row in classMap {
            for label in row {
                labelCounts[label, default: 0] += 1
            }
        }
        print("[Stage0] Label distribution: \(labelCounts.sorted { $0.key < $1.key })")
        
        let rotatedClassMap = rotateClassMap(classMap, by: rotationCount)
        
        var keypoints: [ECGKeypoint] = []
        
        for label in ECGConstants.keypointLabels {
            let mask = rotatedClassMap.map { row in row.map { $0 == label } }
            let (_, stats) = ConnectedComponents.labelAndStatistics(mask: mask)
            
            if let largest = stats.first, largest.area >= ECGConstants.minComponentArea {
                let x = largest.centroidX / scale
                let y = largest.centroidY / scale
                keypoints.append(ECGKeypoint(x: x, y: y, label: label))
                print("[Stage0] Keypoint \(label): (\(x), \(y)) area=\(largest.area)")
            } else {
                // Missing keypoint
                keypoints.append(ECGKeypoint(x: 0, y: 0, label: label))
                let pixelCount = labelCounts[label] ?? 0
                print("[Stage0] Keypoint \(label): NOT FOUND (pixels with this label: \(pixelCount))")
            }
        }
        
        return keypoints
    }
    
    private func rotateClassMap(_ map: [[Int]], by rotationCount: Int) -> [[Int]] {
        let count = ((rotationCount % 4) + 4) % 4
        if count == 0 { return map }
        
        var result = map
        for _ in 0..<count {
            result = rotate90CCW(result)
        }
        return result
    }
    
    private func rotate90CCW<T>(_ matrix: [[T]]) -> [[T]] {
        let rows = matrix.count
        guard rows > 0 else { return matrix }
        let cols = matrix[0].count
        
        var result = [[T]]()
        for c in (0..<cols).reversed() {
            var newRow = [T]()
            for r in 0..<rows {
                newRow.append(matrix[r][c])
            }
            result.append(newRow)
        }
        return result
    }
}

// MARK: - Error Types

enum ECGError: Error, LocalizedError {
    case preprocessingFailed(String)
    case pixelBufferCreationFailed
    case modelOutputMissing(String)
    case insufficientKeypoints(Int)
    case homographyFailed
    case warpFailed
    case gridExtractionFailed
    case rectificationFailed
    case signalExtractionFailed
    
    var errorDescription: String? {
        switch self {
        case .preprocessingFailed(let msg): return "Preprocessing failed: \(msg)"
        case .pixelBufferCreationFailed: return "Failed to create pixel buffer"
        case .modelOutputMissing(let msg): return "Model output missing: \(msg)"
        case .insufficientKeypoints(let count): return "Not enough keypoints: \(count)"
        case .homographyFailed: return "Homography computation failed"
        case .warpFailed: return "Perspective warp failed"
        case .gridExtractionFailed: return "Grid extraction failed"
        case .rectificationFailed: return "Image rectification failed"
        case .signalExtractionFailed: return "Signal extraction failed"
        }
    }
}
