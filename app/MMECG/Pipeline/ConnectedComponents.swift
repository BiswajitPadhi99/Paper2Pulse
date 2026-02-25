import Foundation
import Accelerate

// MARK: - Connected Components Algorithm
// Equivalent to cc3d.connected_components() and cc3d.statistics() in Python

struct ConnectedComponents {
    
    // MARK: - Component Statistics
    
    struct ComponentStats {
        let label: Int
        let area: Int           // Number of pixels
        let centroidX: Float    // Center X coordinate
        let centroidY: Float    // Center Y coordinate
        let boundingBox: (minX: Int, minY: Int, maxX: Int, maxY: Int)
    }
    
    // MARK: - Union-Find
    
    private class UnionFind {
        var parent: [Int]
        var rank: [Int]
        
        init(size: Int) {
            parent = Array(0..<size)
            rank = Array(repeating: 0, count: size)
        }
        
        func find(_ x: Int) -> Int {
            if parent[x] != x {
                parent[x] = find(parent[x])  // Path compression
            }
            return parent[x]
        }
        
        func union(_ x: Int, _ y: Int) {
            let px = find(x)
            let py = find(y)
            if px == py { return }
            
            if rank[px] < rank[py] {
                parent[px] = py
            } else if rank[px] > rank[py] {
                parent[py] = px
            } else {
                parent[py] = px
                rank[px] += 1
            }
        }
    }
    
    // MARK: - Label Connected Components (8-connectivity)
    
    /// Label connected components in a 2D binary mask
    static func label2D(mask: [[Bool]]) -> [[Int]] {
        let height = mask.count
        guard height > 0 else { return [] }
        let width = mask[0].count
        guard width > 0 else { return [] }
        
        var labels = Array(repeating: Array(repeating: 0, count: width), count: height)
        let uf = UnionFind(size: height * width + 1)
        var nextLabel = 1
        
        // First pass
        for y in 0..<height {
            for x in 0..<width {
                guard mask[y][x] else { continue }
                
                var neighborLabels: [Int] = []
                let neighbors = [(y-1, x-1), (y-1, x), (y-1, x+1), (y, x-1)]
                
                for (ny, nx) in neighbors {
                    if ny >= 0 && ny < height && nx >= 0 && nx < width {
                        if labels[ny][nx] > 0 {
                            neighborLabels.append(labels[ny][nx])
                        }
                    }
                }
                
                if neighborLabels.isEmpty {
                    labels[y][x] = nextLabel
                    nextLabel += 1
                } else {
                    let minLabel = neighborLabels.min()!
                    labels[y][x] = minLabel
                    for nl in neighborLabels {
                        uf.union(minLabel, nl)
                    }
                }
            }
        }
        
        // Second pass
        var labelMap: [Int: Int] = [0: 0]
        var finalLabel = 1
        
        for y in 0..<height {
            for x in 0..<width {
                if labels[y][x] > 0 {
                    let root = uf.find(labels[y][x])
                    if labelMap[root] == nil {
                        labelMap[root] = finalLabel
                        finalLabel += 1
                    }
                    labels[y][x] = labelMap[root]!
                }
            }
        }
        
        return labels
    }
    
    /// Label from float array with threshold
    static func label2D(data: [[Float]], threshold: Float) -> [[Int]] {
        let mask = data.map { row in row.map { $0 > threshold } }
        return label2D(mask: mask)
    }
    
    /// Label where values equal specific value
    static func label2D(data: [[Int]], equalTo value: Int) -> [[Int]] {
        let mask = data.map { row in row.map { $0 == value } }
        return label2D(mask: mask)
    }
    
    // MARK: - Compute Statistics
    
    /// Compute statistics for labeled components
    static func statistics(labels: [[Int]]) -> [ComponentStats] {
        let height = labels.count
        guard height > 0 else { return [] }
        let width = labels[0].count
        guard width > 0 else { return [] }
        
        var maxLabel = 0
        for row in labels {
            for l in row { maxLabel = max(maxLabel, l) }
        }
        if maxLabel == 0 { return [] }
        
        var areas = Array(repeating: 0, count: maxLabel + 1)
        var sumX = Array(repeating: 0.0, count: maxLabel + 1)
        var sumY = Array(repeating: 0.0, count: maxLabel + 1)
        var minX = Array(repeating: Int.max, count: maxLabel + 1)
        var minY = Array(repeating: Int.max, count: maxLabel + 1)
        var maxX = Array(repeating: 0, count: maxLabel + 1)
        var maxY = Array(repeating: 0, count: maxLabel + 1)
        
        for y in 0..<height {
            for x in 0..<width {
                let l = labels[y][x]
                if l > 0 {
                    areas[l] += 1
                    sumX[l] += Double(x)
                    sumY[l] += Double(y)
                    minX[l] = min(minX[l], x)
                    minY[l] = min(minY[l], y)
                    maxX[l] = max(maxX[l], x)
                    maxY[l] = max(maxY[l], y)
                }
            }
        }
        
        var stats: [ComponentStats] = []
        for l in 1...maxLabel {
            if areas[l] > 0 {
                stats.append(ComponentStats(
                    label: l,
                    area: areas[l],
                    centroidX: Float(sumX[l] / Double(areas[l])),
                    centroidY: Float(sumY[l] / Double(areas[l])),
                    boundingBox: (minX[l], minY[l], maxX[l], maxY[l])
                ))
            }
        }
        
        stats.sort { $0.area > $1.area }
        return stats
    }
    
    /// Combined: label and compute statistics
    static func labelAndStatistics(mask: [[Bool]]) -> (labels: [[Int]], stats: [ComponentStats]) {
        let labels = label2D(mask: mask)
        let stats = statistics(labels: labels)
        return (labels, stats)
    }
    
    static func labelAndStatistics(data: [[Float]], threshold: Float) -> (labels: [[Int]], stats: [ComponentStats]) {
        let labels = label2D(data: data, threshold: threshold)
        let stats = statistics(labels: labels)
        return (labels, stats)
    }
}

// MARK: - Array Utilities

extension ConnectedComponents {
    
    /// Extract 2D slice from flattened [C,H,W] array
    static func extractChannel(from data: [Float], channel: Int, height: Int, width: Int) -> [[Float]] {
        var result: [[Float]] = []
        let offset = channel * height * width
        
        for y in 0..<height {
            var row: [Float] = []
            for x in 0..<width {
                row.append(data[offset + y * width + x])
            }
            result.append(row)
        }
        return result
    }
    
    /// Get argmax along channel dimension
    /// Input: [C,H,W] flattened, Output: [H,W] of argmax indices
    /// Note: dataHeight/dataWidth are the actual tensor dimensions (for stride calculation)
    ///       iterHeight/iterWidth are the bounds for iteration (can be smaller)
    static func argmaxAlongChannels(
        data: [Float],
        channels: Int,
        height: Int,
        width: Int,
        dataHeight: Int? = nil,
        dataWidth: Int? = nil
    ) -> [[Int]] {
        // Use actual data dimensions for stride, iteration dimensions for bounds
        let strideH = dataHeight ?? height
        let strideW = dataWidth ?? width
        
        var result = Array(repeating: Array(repeating: 0, count: width), count: height)
        
        for y in 0..<height {
            for x in 0..<width {
                var maxVal: Float = -.infinity
                var maxIdx = 0
                
                for c in 0..<channels {
                    // Use strideH and strideW for correct memory layout
                    let idx = c * strideH * strideW + y * strideW + x
                    if idx < data.count && data[idx] > maxVal {
                        maxVal = data[idx]
                        maxIdx = c
                    }
                }
                result[y][x] = maxIdx
            }
        }
        return result
    }
    
    /// Get max value along channel dimension
    static func maxAlongChannels(data: [Float], channels: Int, height: Int, width: Int) -> [[Float]] {
        var result = Array(repeating: Array(repeating: Float(0), count: width), count: height)
        
        for y in 0..<height {
            for x in 0..<width {
                var maxVal: Float = -.infinity
                for c in 0..<channels {
                    let idx = c * height * width + y * width + x
                    maxVal = max(maxVal, data[idx])
                }
                result[y][x] = maxVal
            }
        }
        return result
    }
}
