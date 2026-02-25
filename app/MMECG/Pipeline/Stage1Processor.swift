import UIKit
import CoreML

// MARK: - Stage 1 Processor
// Handles: Normalized image → Model inference → Grid extraction → Rectification

class Stage1Processor {
    
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
    
    struct Stage1Result {
        let rectifiedImage: UIImage
        let gridPointsXY: [[[Float]]]?
        let usedFallback: Bool
    }
    
    /// Process normalized image through Stage 1
    func process(normalizedImage: UIImage) throws -> Stage1Result {
        print("[Stage1] Starting processing...")
        print("[Stage1] Input size: \(normalizedImage.size)")
        
        // 1. Convert to pixel buffer
        guard let pixelBuffer = ImageUtils.pixelBuffer(
            from: normalizedImage,
            width: ECGConstants.normalizedWidth,
            height: ECGConstants.normalizedHeight
        ) else {
            throw ECGError.pixelBufferCreationFailed
        }
        
        // 2. Run model inference
        print("[Stage1] Running model inference...")
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
        let output = try model.prediction(from: input)
        
        // 3. Extract outputs
        guard let gridpointArray = output.featureValue(for: "gridpoint")?.multiArrayValue,
              let gridhlineArray = output.featureValue(for: "gridhline")?.multiArrayValue,
              let gridvlineArray = output.featureValue(for: "gridvline")?.multiArrayValue else {
            throw ECGError.modelOutputMissing("gridpoint/gridhline/gridvline missing")
        }
        
        print("[Stage1] gridpoint shape: \(gridpointArray.shapeArray)")
        print("[Stage1] gridhline shape: \(gridhlineArray.shapeArray)")
        print("[Stage1] gridvline shape: \(gridvlineArray.shapeArray)")
        
        let gridpointData = gridpointArray.toFloatArray()
        let gridhlineData = gridhlineArray.toFloatArray()
        let gridvlineData = gridvlineArray.toFloatArray()
        
        // 4. Try to extract grid points
        var gridXY: [[[Float]]]? = nil
        var rectifiedImage: UIImage
        var usedFallback = false
        
        do {
            gridXY = try extractGridPoints(
                gridpointData: gridpointData,
                gridhlineData: gridhlineData,
                gridvlineData: gridvlineData,
                height: ECGConstants.normalizedHeight,
                width: ECGConstants.normalizedWidth
            )
            
            // Check if we have enough valid grid points
            let validCount = countValidGridPoints(gridXY!)
            print("[Stage1] Valid grid points: \(validCount)")
            
            if validCount > 100 {
                // Interpolate missing points
                GridSampler.interpolateMissingGridPoints(gridXY: &gridXY!)
                
                // Rectify image
                if let rectified = GridSampler.rectifyImage(
                    image: normalizedImage,
                    gridPointsXY: gridXY!,
                    outputSize: CGSize(
                        width: ECGConstants.rectifiedWidth,
                        height: ECGConstants.rectifiedHeight
                    )
                ) {
                    rectifiedImage = rectified
                    print("[Stage1] Rectification successful")
                } else {
                    throw ECGError.rectificationFailed
                }
            } else {
                throw ECGError.gridExtractionFailed
            }
        } catch {
            print("[Stage1] Grid extraction/rectification failed: \(error), using fallback")
            rectifiedImage = fallbackRectify(image: normalizedImage)
            usedFallback = true
        }
        
        return Stage1Result(
            rectifiedImage: rectifiedImage,
            gridPointsXY: gridXY,
            usedFallback: usedFallback
        )
    }
    
    /// Fallback: simple resize without grid rectification
    private func fallbackRectify(image: UIImage) -> UIImage {
        let targetSize = CGSize(width: ECGConstants.rectifiedWidth,
                                height: ECGConstants.rectifiedHeight)
        return ImageUtils.resize(image: image, to: targetSize) ?? image
    }
    
    /// Count valid (non-zero) grid points
    private func countValidGridPoints(_ grid: [[[Float]]]) -> Int {
        var count = 0
        for row in grid {
            for point in row {
                if point[0] != 0 || point[1] != 0 {
                    count += 1
                }
            }
        }
        return count
    }
    
    // MARK: - Grid Point Extraction
    
    private func extractGridPoints(
        gridpointData: [Float],
        gridhlineData: [Float],
        gridvlineData: [Float],
        height: Int,
        width: Int
    ) throws -> [[[Float]]] {
        
        let numHLines = ECGConstants.gridRows + 1  // 45
        let numVLines = ECGConstants.gridCols + 1  // 58
        
        // 1. Get gridpoint centroids
        let gridpointChannel = extractChannel2D(gridpointData, height: height, width: width)
        let (_, pointStats) = ConnectedComponents.labelAndStatistics(
            data: gridpointChannel,
            threshold: ECGConstants.gridpointThreshold
        )
        
        print("[Stage1] Found \(pointStats.count) grid point candidates")
        
        // 2. Get line class maps
        let vlineClassMap = ConnectedComponents.argmaxAlongChannels(
            data: gridvlineData,
            channels: numVLines,
            height: height,
            width: width
        )
        
        let hlineClassMap = ConnectedComponents.argmaxAlongChannels(
            data: gridhlineData,
            channels: numHLines,
            height: height,
            width: width
        )
        
        // 3. Build grid mapping
        var gridXY = [[[Float]]](
            repeating: [[Float]](repeating: [Float](repeating: 0, count: 2), count: ECGConstants.gridCols),
            count: ECGConstants.gridRows
        )
        
        for stat in pointStats {
            let x = Int(round(stat.centroidX))
            let y = Int(round(stat.centroidY))
            
            guard x >= 0 && x < width && y >= 0 && y < height else { continue }
            
            let vIdx = vlineClassMap[y][x]
            let hIdx = hlineClassMap[y][x]
            
            // Line indices are 1-based (0 = background)
            if vIdx > 0 && hIdx > 0 && vIdx <= ECGConstants.gridCols && hIdx <= ECGConstants.gridRows {
                gridXY[hIdx - 1][vIdx - 1][0] = stat.centroidX
                gridXY[hIdx - 1][vIdx - 1][1] = stat.centroidY
            }
        }
        
        return gridXY
    }
    
    private func extractChannel2D(_ data: [Float], height: Int, width: Int) -> [[Float]] {
        var result = [[Float]]()
        for y in 0..<height {
            var row = [Float]()
            for x in 0..<width {
                row.append(data[y * width + x])
            }
            result.append(row)
        }
        return result
    }
}
