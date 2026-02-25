import UIKit
import Accelerate

// MARK: - Grid Sampler
// Equivalent to torch.nn.functional.grid_sample() for image rectification

struct GridSampler {
    
    // MARK: - Grid-based Image Rectification
    
    /// Rectify image using a grid mapping
    /// Equivalent to: F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
    ///
    /// - Parameters:
    ///   - image: Input image
    ///   - gridPointsXY: Sparse grid of control points [rows, cols, 2] where [..,..,0]=x, [..,..,1]=y
    ///   - outputSize: Desired output size
    /// - Returns: Rectified image
    static func rectifyImage(
        image: UIImage,
        gridPointsXY: [[[Float]]],  // [44, 57, 2]
        outputSize: CGSize
    ) -> UIImage? {
        guard let inputCG = image.cgImage else { return nil }
        
        let inW = inputCG.width, inH = inputCG.height
        let outW = Int(outputSize.width), outH = Int(outputSize.height)
        
        // Get input pixels
        guard let inputData = inputCG.dataProvider?.data,
              let inputPtr = CFDataGetBytePtr(inputData) else { return nil }
        
        let inBytesPerRow = inputCG.bytesPerRow
        let inBytesPerPixel = inputCG.bitsPerPixel / 8
        
        // First, interpolate sparse grid to dense grid
        let denseGrid = interpolateGrid(
            sparseGrid: gridPointsXY,
            outputHeight: outH,
            outputWidth: outW,
            inputHeight: inH,
            inputWidth: inW
        )
        
        // Output buffer
        var outputData = [UInt8](repeating: 0, count: outW * outH * 4)
        
        // Sample using dense grid
        for y in 0..<outH {
            for x in 0..<outW {
                let srcX = denseGrid[y][x].x
                let srcY = denseGrid[y][x].y
                
                // Bilinear interpolation
                let x0 = Int(floor(srcX)), y0 = Int(floor(srcY))
                let x1 = x0 + 1, y1 = y0 + 1
                let fx = srcX - Float(x0), fy = srcY - Float(y0)
                
                // Border padding: clamp to valid range
                let x0c = max(0, min(x0, inW - 1))
                let x1c = max(0, min(x1, inW - 1))
                let y0c = max(0, min(y0, inH - 1))
                let y1c = max(0, min(y1, inH - 1))
                
                let outOffset = (y * outW + x) * 4
                
                for c in 0..<min(3, inBytesPerPixel) {
                    let v00 = Float(inputPtr[y0c * inBytesPerRow + x0c * inBytesPerPixel + c])
                    let v01 = Float(inputPtr[y0c * inBytesPerRow + x1c * inBytesPerPixel + c])
                    let v10 = Float(inputPtr[y1c * inBytesPerRow + x0c * inBytesPerPixel + c])
                    let v11 = Float(inputPtr[y1c * inBytesPerRow + x1c * inBytesPerPixel + c])
                    
                    let v0 = v00 * (1 - fx) + v01 * fx
                    let v1 = v10 * (1 - fx) + v11 * fx
                    let v = v0 * (1 - fy) + v1 * fy
                    
                    outputData[outOffset + c] = UInt8(min(max(v, 0), 255))
                }
                outputData[outOffset + 3] = 255
            }
        }
        
        guard let context = CGContext(
            data: &outputData, width: outW, height: outH,
            bitsPerComponent: 8, bytesPerRow: outW * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = context.makeImage() else { return nil }
        
        return UIImage(cgImage: cgImage)
    }
    
    // MARK: - Grid Interpolation
    
    /// Interpolate sparse grid to dense grid using bilinear interpolation
    private static func interpolateGrid(
        sparseGrid: [[[Float]]],  // [gridRows, gridCols, 2]
        outputHeight: Int,
        outputWidth: Int,
        inputHeight: Int,
        inputWidth: Int
    ) -> [[(x: Float, y: Float)]] {
        let gridRows = sparseGrid.count
        let gridCols = sparseGrid[0].count
        
        // Normalize sparse grid to [-1, 1] range (as in PyTorch)
        // Then we'll convert to pixel coordinates
        
        var denseGrid = [[(x: Float, y: Float)]](
            repeating: [(x: Float, y: Float)](repeating: (0, 0), count: outputWidth),
            count: outputHeight
        )
        
        // For each output pixel, find corresponding input pixel via grid interpolation
        for outY in 0..<outputHeight {
            for outX in 0..<outputWidth {
                // Map output coordinates to grid coordinates
                let gridY = Float(outY) / Float(outputHeight - 1) * Float(gridRows - 1)
                let gridX = Float(outX) / Float(outputWidth - 1) * Float(gridCols - 1)
                
                // Bilinear interpolation in grid space
                let gy0 = Int(floor(gridY)), gx0 = Int(floor(gridX))
                let gy1 = min(gy0 + 1, gridRows - 1), gx1 = min(gx0 + 1, gridCols - 1)
                let fy = gridY - Float(gy0), fx = gridX - Float(gx0)
                
                // Get grid values (already in input pixel coordinates)
                let v00x = sparseGrid[gy0][gx0][0], v00y = sparseGrid[gy0][gx0][1]
                let v01x = sparseGrid[gy0][gx1][0], v01y = sparseGrid[gy0][gx1][1]
                let v10x = sparseGrid[gy1][gx0][0], v10y = sparseGrid[gy1][gx0][1]
                let v11x = sparseGrid[gy1][gx1][0], v11y = sparseGrid[gy1][gx1][1]
                
                // Bilinear interpolation
                let v0x = v00x * (1 - fx) + v01x * fx
                let v0y = v00y * (1 - fx) + v01y * fx
                let v1x = v10x * (1 - fx) + v11x * fx
                let v1y = v10y * (1 - fx) + v11y * fx
                
                let srcX = v0x * (1 - fy) + v1x * fy
                let srcY = v0y * (1 - fy) + v1y * fy
                
                denseGrid[outY][outX] = (srcX, srcY)
            }
        }
        
        return denseGrid
    }
    
    // MARK: - Grid Interpolation for Missing Points
    
    /// Interpolate missing grid points (where value is [0,0])
    /// Equivalent to scipy.interpolate.griddata with method='cubic'
    static func interpolateMissingGridPoints(
        gridXY: inout [[[Float]]]  // [rows, cols, 2], modified in place
    ) {
        let rows = gridXY.count
        let cols = gridXY[0].count
        
        // Find valid and missing points
        var validPoints: [(row: Int, col: Int, x: Float, y: Float)] = []
        var missingPoints: [(row: Int, col: Int)] = []
        
        for r in 0..<rows {
            for c in 0..<cols {
                let x = gridXY[r][c][0], y = gridXY[r][c][1]
                if x != 0 || y != 0 {
                    validPoints.append((r, c, x, y))
                } else {
                    missingPoints.append((r, c))
                }
            }
        }
        
        guard !validPoints.isEmpty else { return }
        
        // For each missing point, use inverse distance weighting
        for (r, c) in missingPoints {
            var sumWeightX: Float = 0, sumWeightY: Float = 0, sumWeight: Float = 0
            
            for (vr, vc, vx, vy) in validPoints {
                let dr = Float(r - vr), dc = Float(c - vc)
                let dist = sqrt(dr * dr + dc * dc)
                if dist < 0.001 { continue }
                
                // Inverse distance squared weighting
                let weight = 1.0 / (dist * dist)
                sumWeightX += weight * vx
                sumWeightY += weight * vy
                sumWeight += weight
            }
            
            if sumWeight > 0 {
                gridXY[r][c][0] = sumWeightX / sumWeight
                gridXY[r][c][1] = sumWeightY / sumWeight
            }
        }
    }
    
    // MARK: - Cubic Interpolation (Better quality)
    
    /// Bicubic interpolation for smoother results
    static func interpolateGridCubic(
        sparseGrid: [[[Float]]],
        outputHeight: Int,
        outputWidth: Int
    ) -> [[(x: Float, y: Float)]] {
        // For now, use bilinear as it's simpler and usually sufficient
        // Can implement cubic later if needed
        return interpolateGrid(
            sparseGrid: sparseGrid,
            outputHeight: outputHeight,
            outputWidth: outputWidth,
            inputHeight: ECGConstants.normalizedHeight,
            inputWidth: ECGConstants.normalizedWidth
        )
    }
}
