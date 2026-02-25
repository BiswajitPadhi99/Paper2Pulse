import UIKit
import CoreML

// MARK: - Debug Diagnostics for Stage 0
// Add this to your project to diagnose keypoint detection issues

class Stage0Diagnostics {
    
    /// Run full diagnostics on Stage 0 processing
    static func diagnose(image: UIImage, modelURL: URL) {
        print("=" * 60)
        print("STAGE 0 DIAGNOSTICS")
        print("=" * 60)
        
        // 1. Check input image
        print("\n[1] INPUT IMAGE")
        print("  Size: \(image.size)")
        print("  Scale: \(image.scale)")
        print("  Orientation: \(image.imageOrientation.rawValue)")
        
        // 2. Preprocess
        print("\n[2] PREPROCESSING")
        guard let preprocessed = ImageUtils.resizeAndPad(
            image: image,
            targetWidth: 1440,
            padToMultiple: 32
        ) else {
            print("  ERROR: Preprocessing failed")
            return
        }
        
        let paddedImage = preprocessed.image
        print("  Original: \(image.size.width) x \(image.size.height)")
        print("  Scale factor: \(preprocessed.scale)")
        print("  Scaled height: \(preprocessed.scaledHeight)")
        print("  Padded size: \(preprocessed.paddedSize)")
        
        // 3. Save preprocessed image for visual inspection
        if let data = paddedImage.pngData() {
            let path = FileManager.default.temporaryDirectory.appendingPathComponent("stage0_input.png")
            try? data.write(to: path)
            print("  Saved preprocessed image to: \(path)")
        }
        
        // 4. Create pixel buffer
        print("\n[3] PIXEL BUFFER")
        let paddedWidth = Int(preprocessed.paddedSize.width)
        let paddedHeight = Int(preprocessed.paddedSize.height)
        
        guard let pixelBuffer = ImageUtils.pixelBuffer(
            from: paddedImage,
            width: paddedWidth,
            height: paddedHeight
        ) else {
            print("  ERROR: Pixel buffer creation failed")
            return
        }
        
        print("  Buffer size: \(CVPixelBufferGetWidth(pixelBuffer)) x \(CVPixelBufferGetHeight(pixelBuffer))")
        print("  Pixel format: \(CVPixelBufferGetPixelFormatType(pixelBuffer))")
        
        // Check pixel values
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        if let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) {
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
            
            // Sample pixels at different locations
            print("\n[4] PIXEL SAMPLES (BGRA format)")
            let samplePoints = [
                (0, 0, "Top-left"),
                (paddedWidth/2, 0, "Top-center"),
                (paddedWidth-1, 0, "Top-right"),
                (0, paddedHeight/2, "Middle-left"),
                (paddedWidth/2, paddedHeight/2, "Center"),
                (0, paddedHeight-1, "Bottom-left"),
                (paddedWidth/2, paddedHeight-1, "Bottom-center")
            ]
            
            for (x, y, label) in samplePoints {
                let offset = y * bytesPerRow + x * 4
                let b = ptr[offset]
                let g = ptr[offset + 1]
                let r = ptr[offset + 2]
                let a = ptr[offset + 3]
                print("  \(label) (\(x), \(y)): R=\(r) G=\(g) B=\(b) A=\(a)")
            }
        }
        
        // 5. Run model
        print("\n[5] MODEL INFERENCE")
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
            let output = try model.prediction(from: input)
            
            guard let markerArray = output.featureValue(for: "marker")?.multiArrayValue,
                  let orientationArray = output.featureValue(for: "orientation")?.multiArrayValue else {
                print("  ERROR: Model output missing")
                return
            }
            
            print("  Marker shape: \(markerArray.shapeArray)")
            print("  Orientation shape: \(orientationArray.shapeArray)")
            
            // 6. Analyze marker output
            print("\n[6] MARKER OUTPUT ANALYSIS")
            let markerData = markerArray.toFloatArray()
            let channels = markerArray.shapeArray[1]
            let height = markerArray.shapeArray[2]
            let width = markerArray.shapeArray[3]
            
            // Check min/max per channel
            print("  Channel statistics:")
            for c in 0..<channels {
                var minVal: Float = .infinity
                var maxVal: Float = -.infinity
                var sum: Float = 0
                
                for y in 0..<height {
                    for x in 0..<width {
                        let idx = c * height * width + y * width + x
                        let val = markerData[idx]
                        minVal = min(minVal, val)
                        maxVal = max(maxVal, val)
                        sum += val
                    }
                }
                let mean = sum / Float(height * width)
                print("    Ch \(c): min=\(String(format: "%.4f", minVal)), max=\(String(format: "%.4f", maxVal)), mean=\(String(format: "%.6f", mean))")
            }
            
            // 7. Check argmax distribution
            print("\n[7] ARGMAX DISTRIBUTION")
            var labelCounts = [Int: Int]()
            
            for y in 0..<height {
                for x in 0..<width {
                    var maxVal: Float = -.infinity
                    var maxIdx = 0
                    
                    for c in 0..<channels {
                        let idx = c * height * width + y * width + x
                        if markerData[idx] > maxVal {
                            maxVal = markerData[idx]
                            maxIdx = c
                        }
                    }
                    labelCounts[maxIdx, default: 0] += 1
                }
            }
            
            print("  Label distribution:")
            for label in labelCounts.keys.sorted() {
                let count = labelCounts[label]!
                let pct = Float(count) / Float(height * width) * 100
                print("    Label \(label): \(count) pixels (\(String(format: "%.2f", pct))%)")
            }
            
            // 8. Check specific rows for keypoint labels
            print("\n[8] KEYPOINT LABEL LOCATIONS")
            let keypointLabels = [2, 3, 4, 6, 7, 8, 10, 11, 12]
            
            for label in keypointLabels {
                var foundLocations: [(Int, Int, Float)] = []
                
                for y in 0..<height {
                    for x in 0..<width {
                        var maxVal: Float = -.infinity
                        var maxIdx = 0
                        
                        for c in 0..<channels {
                            let idx = c * height * width + y * width + x
                            if markerData[idx] > maxVal {
                                maxVal = markerData[idx]
                                maxIdx = c
                            }
                        }
                        
                        if maxIdx == label {
                            foundLocations.append((x, y, maxVal))
                        }
                    }
                }
                
                if foundLocations.isEmpty {
                    // Check what the model outputs at expected keypoint location
                    let expectedY = [350, 350, 350, 500, 500, 500, 650, 650, 650]
                    let expectedX = [450, 750, 1050, 450, 750, 1050, 450, 750, 1050]
                    let idx = keypointLabels.firstIndex(of: label)!
                    let ey = expectedY[idx]
                    let ex = expectedX[idx]
                    
                    print("    Label \(label): NOT FOUND")
                    print("      At expected location (\(ex), \(ey)):")
                    
                    var probs: [(Int, Float)] = []
                    for c in 0..<channels {
                        let dataIdx = c * height * width + ey * width + ex
                        if dataIdx < markerData.count {
                            probs.append((c, markerData[dataIdx]))
                        }
                    }
                    probs.sort { $0.1 > $1.1 }
                    for i in 0..<min(3, probs.count) {
                        print("        Ch \(probs[i].0): \(String(format: "%.4f", probs[i].1))")
                    }
                } else {
                    let minY = foundLocations.map { $0.1 }.min()!
                    let maxY = foundLocations.map { $0.1 }.max()!
                    let minX = foundLocations.map { $0.0 }.min()!
                    let maxX = foundLocations.map { $0.0 }.max()!
                    print("    Label \(label): \(foundLocations.count) pixels, bbox=(\(minX),\(minY))-(\(maxX),\(maxY))")
                }
            }
            
            // 9. Orientation output
            print("\n[9] ORIENTATION OUTPUT")
            let orientationData = orientationArray.toFloatArray()
            print("  Probabilities: ", terminator: "")
            for (i, val) in orientationData.enumerated() {
                print("\(i):\(String(format: "%.3f", val)) ", terminator: "")
            }
            print()
            
        } catch {
            print("  ERROR: \(error)")
        }
        
        print("\n" + "=" * 60)
    }
    
    /// Test with flipped image to diagnose Y-axis issues
    static func diagnoseWithFlip(image: UIImage, modelURL: URL) {
        print("\n\n")
        print("=" * 60)
        print("TESTING WITH FLIPPED IMAGE")
        print("=" * 60)
        
        // Flip vertically
        guard let cgImage = image.cgImage else { return }
        let flippedImage = UIImage(cgImage: cgImage, scale: image.scale, orientation: .downMirrored)
        
        diagnose(image: flippedImage, modelURL: modelURL)
    }
}

// Helper extension
private func * (string: String, times: Int) -> String {
    return String(repeating: string, count: times)
}

// MARK: - Alternative Pixel Buffer Creation (No Flip)
// Try this if the standard method has coordinate issues

extension ImageUtils {
    
    /// Alternative pixel buffer creation without coordinate flipping
    static func pixelBufferNoFlip(from image: UIImage, width: Int, height: Int) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else { return nil }
        
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }
        
        // Draw without flipping - CGContext origin is bottom-left by default
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return buffer
    }
    
    /// Pixel buffer using VImage for guaranteed correct conversion
    static func pixelBufferVImage(from image: UIImage, width: Int, height: Int) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else { return nil }
        
        // Create destination pixel buffer
        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        
        CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pixelBuffer
        )
        
        guard let buffer = pixelBuffer else { return nil }
        
        // Use CIContext for conversion
        let ciImage = CIImage(cgImage: cgImage)
        let ciContext = CIContext(options: nil)
        ciContext.render(ciImage, to: buffer)
        
        return buffer
    }
}

