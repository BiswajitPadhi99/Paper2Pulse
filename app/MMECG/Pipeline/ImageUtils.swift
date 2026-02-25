import UIKit
import CoreImage
import Accelerate
import CoreML

// MARK: - Image Utilities

class ImageUtils {
    
    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    
    // MARK: - UIImage ↔ CVPixelBuffer
    
    /// Convert UIImage to CVPixelBuffer for CoreML (RGB format)
    /// Note: CoreML ImageType with RGB color_layout handles BGRA→RGB conversion
    static func pixelBuffer(from image: UIImage, width: Int, height: Int) -> CVPixelBuffer? {
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
        
        // Draw CGImage directly - CGContext expects origin at bottom-left,
        // but CGImage also has origin at bottom-left, so they match
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return buffer
    }
    
    /// Convert CVPixelBuffer to UIImage
    static func image(from pixelBuffer: CVPixelBuffer) -> UIImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        return UIImage(cgImage: cgImage)
    }
    
    // MARK: - Resize & Pad
    
    /// Resize image to width, maintaining aspect ratio, pad to multiple of 32
    static func resizeAndPad(
        image: UIImage,
        targetWidth: Int,
        padToMultiple: Int = 32
    ) -> (image: UIImage, scale: Float, scaledHeight: Int, paddedSize: CGSize)? {
        let originalSize = image.size
        let scale = Float(targetWidth) / Float(originalSize.width)
        
        let scaledWidth = targetWidth
        let scaledHeight = Int(Float(originalSize.height) * scale)
        
        let paddedWidth = ((scaledWidth / padToMultiple) + 1) * padToMultiple
        let paddedHeight = ((scaledHeight / padToMultiple) + 1) * padToMultiple
        
        UIGraphicsBeginImageContextWithOptions(CGSize(width: paddedWidth, height: paddedHeight), true, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        guard let context = UIGraphicsGetCurrentContext() else { return nil }
        
        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: paddedWidth, height: paddedHeight))
        
        image.draw(in: CGRect(x: 0, y: 0, width: scaledWidth, height: scaledHeight))
        
        guard let paddedImage = UIGraphicsGetImageFromCurrentImageContext() else { return nil }
        
        return (paddedImage, scale, scaledHeight, CGSize(width: paddedWidth, height: paddedHeight))
    }
    
    /// Simple resize to exact dimensions
    static func resize(image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        defer { UIGraphicsEndImageContext() }
        image.draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    // MARK: - Rotate Image
    
    /// Rotate image by 90° increments (rotationCount: number of CCW 90° rotations)
    static func rotate(image: UIImage, by rotationCount: Int) -> UIImage {
        let count = ((rotationCount % 4) + 4) % 4
        if count == 0 { return image }
        
        guard let cgImage = image.cgImage else { return image }
        
        let width = cgImage.width, height = cgImage.height
        let newWidth = (count == 1 || count == 3) ? height : width
        let newHeight = (count == 1 || count == 3) ? width : height
        
        guard let context = CGContext(
            data: nil, width: newWidth, height: newHeight,
            bitsPerComponent: cgImage.bitsPerComponent, bytesPerRow: 0,
            space: cgImage.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: cgImage.bitmapInfo.rawValue
        ) else { return image }
        
        context.translateBy(x: CGFloat(newWidth) / 2, y: CGFloat(newHeight) / 2)
        context.rotate(by: CGFloat(count) * .pi / 2)
        context.translateBy(x: -CGFloat(width) / 2, y: -CGFloat(height) / 2)
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let rotated = context.makeImage() else { return image }
        return UIImage(cgImage: rotated)
    }
    
    // MARK: - Perspective Warp
    
    /// Apply homography (perspective transform) to image
    static func warpPerspective(
        image: UIImage,
        homography: [[Double]],
        outputSize: CGSize
    ) -> UIImage? {
        guard let inputCG = image.cgImage else { return nil }
        
        let outW = Int(outputSize.width), outH = Int(outputSize.height)
        let inW = inputCG.width, inH = inputCG.height
        
        print("[Warp] Input size: \(inW) x \(inH)")
        print("[Warp] Output size: \(outW) x \(outH)")
        print("[Warp] Homography H:")
        for row in homography {
            print("[Warp]   [\(row.map { String(format: "%15.8e", $0) }.joined(separator: ", "))]")
        }
        
        // Get input pixels
        guard let inputData = inputCG.dataProvider?.data,
              let inputPtr = CFDataGetBytePtr(inputData) else { return nil }
        
        let inBytesPerRow = inputCG.bytesPerRow
        let inBytesPerPixel = inputCG.bitsPerPixel / 8
        
        print("[Warp] Input bytesPerRow: \(inBytesPerRow), bytesPerPixel: \(inBytesPerPixel)")
        
        // Inverse homography for backward mapping
        let invH = invert3x3(homography)
        
        print("[Warp] Inverse Homography H^-1:")
        for row in invH {
            print("[Warp]   [\(row.map { String(format: "%15.8e", $0) }.joined(separator: ", "))]")
        }
        
        // Test corner mappings
        let testPoints = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: outW-1, y: 0),
            CGPoint(x: 0, y: outH-1),
            CGPoint(x: outW-1, y: outH-1),
            CGPoint(x: outW/2, y: outH/2)
        ]
        print("[Warp] Output -> Source mappings (using H^-1):")
        for pt in testPoints {
            let srcPt = applyHomography(invH, to: pt)
            let inBounds = srcPt.x >= 0 && srcPt.x < CGFloat(inW) && srcPt.y >= 0 && srcPt.y < CGFloat(inH)
            print("[Warp]   out(\(Int(pt.x)), \(Int(pt.y))) -> src(\(String(format: "%.1f", srcPt.x)), \(String(format: "%.1f", srcPt.y))) \(inBounds ? "✓" : "OUT OF BOUNDS")")
        }
        
        // Output buffer
        var outputData = [UInt8](repeating: 0, count: outW * outH * 4)
        var validPixels = 0
        
        // Backward mapping with bilinear interpolation
        for y in 0..<outH {
            for x in 0..<outW {
                let srcPt = applyHomography(invH, to: CGPoint(x: x, y: y))
                let sx = Float(srcPt.x), sy = Float(srcPt.y)
                
                // Bilinear interpolation
                let x0 = Int(floor(sx)), y0 = Int(floor(sy))
                let x1 = x0 + 1, y1 = y0 + 1
                let fx = sx - Float(x0), fy = sy - Float(y0)
                
                if x0 >= 0 && x1 < inW && y0 >= 0 && y1 < inH {
                    let outOffset = (y * outW + x) * 4
                    
                    for c in 0..<3 {
                        let v00 = Float(inputPtr[y0 * inBytesPerRow + x0 * inBytesPerPixel + c])
                        let v01 = Float(inputPtr[y0 * inBytesPerRow + x1 * inBytesPerPixel + c])
                        let v10 = Float(inputPtr[y1 * inBytesPerRow + x0 * inBytesPerPixel + c])
                        let v11 = Float(inputPtr[y1 * inBytesPerRow + x1 * inBytesPerPixel + c])
                        
                        let v0 = v00 * (1 - fx) + v01 * fx
                        let v1 = v10 * (1 - fx) + v11 * fx
                        let v = v0 * (1 - fy) + v1 * fy
                        
                        outputData[outOffset + c] = UInt8(min(max(v, 0), 255))
                    }
                    outputData[outOffset + 3] = 255
                    validPixels += 1
                }
            }
        }
        
        print("[Warp] Valid pixels: \(validPixels) / \(outW * outH) (\(String(format: "%.1f", Double(validPixels) / Double(outW * outH) * 100))%)")
        
        guard let context = CGContext(
            data: &outputData, width: outW, height: outH,
            bitsPerComponent: 8, bytesPerRow: outW * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = context.makeImage() else { return nil }
        
        return UIImage(cgImage: cgImage)
    }
    
    private static func invert3x3(_ m: [[Double]]) -> [[Double]] {
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        guard abs(det) > 1e-10 else { return [[1,0,0], [0,1,0], [0,0,1]] }
        let inv = 1.0 / det
        return [
            [inv*(m[1][1]*m[2][2]-m[1][2]*m[2][1]), inv*(m[0][2]*m[2][1]-m[0][1]*m[2][2]), inv*(m[0][1]*m[1][2]-m[0][2]*m[1][1])],
            [inv*(m[1][2]*m[2][0]-m[1][0]*m[2][2]), inv*(m[0][0]*m[2][2]-m[0][2]*m[2][0]), inv*(m[0][2]*m[1][0]-m[0][0]*m[1][2])],
            [inv*(m[1][0]*m[2][1]-m[1][1]*m[2][0]), inv*(m[0][1]*m[2][0]-m[0][0]*m[2][1]), inv*(m[0][0]*m[1][1]-m[0][1]*m[1][0])]
        ]
    }
    
    private static func applyHomography(_ H: [[Double]], to p: CGPoint) -> CGPoint {
        let x = Double(p.x), y = Double(p.y)
        let w = H[2][0] * x + H[2][1] * y + H[2][2]
        guard abs(w) > 1e-10 else { return p }
        return CGPoint(x: (H[0][0]*x + H[0][1]*y + H[0][2]) / w,
                       y: (H[1][0]*x + H[1][1]*y + H[1][2]) / w)
    }
    
    // MARK: - Crop
    
    static func crop(image: UIImage, to rect: CGRect) -> UIImage? {
        guard let cgImage = image.cgImage,
              let cropped = cgImage.cropping(to: rect) else { return nil }
        return UIImage(cgImage: cropped)
    }
}

// MARK: - MLMultiArray Extension

extension MLMultiArray {
    var shapeArray: [Int] { shape.map { $0.intValue } }
    
    func toFloatArray() -> [Float] {
        let count = shape.reduce(1) { $0 * $1.intValue }
        let ptr = UnsafeMutablePointer<Float>(OpaquePointer(dataPointer))
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
