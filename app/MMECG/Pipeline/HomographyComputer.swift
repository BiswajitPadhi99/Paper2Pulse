
import Foundation
import Accelerate
import CoreGraphics

// MARK: - Homography Computer
// Equivalent to cv2.findHomography() with RANSAC

struct HomographyComputer {
    
    static var debugEnabled = true
    
    private static func debugLog(_ message: String) {
        if debugEnabled {
            print("[Homography] \(message)")
        }
    }
    
    // MARK: - Compute Homography
    
    /// Compute homography from source to destination points
    /// Returns 3x3 matrix and inlier mask
    static func findHomography(
        srcPoints: [CGPoint],
        dstPoints: [CGPoint],
        useRANSAC: Bool = true,
        threshold: CGFloat = 5.0
    ) -> (matrix: [[Double]], inlierMask: [Bool])? {
        
        guard srcPoints.count >= 4, srcPoints.count == dstPoints.count else {
            debugLog("ERROR: Need at least 4 points, got \(srcPoints.count)")
            return nil
        }
        
        debugLog("=== findHomography ===")
        debugLog("Number of points: \(srcPoints.count)")
        debugLog("Source points (detected keypoints):")
        for (i, p) in srcPoints.enumerated() {
            debugLog("  src[\(i)]: (\(String(format: "%.2f", p.x)), \(String(format: "%.2f", p.y)))")
        }
        debugLog("Destination points (reference):")
        for (i, p) in dstPoints.enumerated() {
            debugLog("  dst[\(i)]: (\(String(format: "%.2f", p.x)), \(String(format: "%.2f", p.y)))")
        }
        
        if useRANSAC && srcPoints.count > 4 {
            return ransac(srcPoints: srcPoints, dstPoints: dstPoints, threshold: threshold)
        } else {
            guard let H = dlt(srcPoints: srcPoints, dstPoints: dstPoints) else { return nil }
            return (H, Array(repeating: true, count: srcPoints.count))
        }
    }
    
    // MARK: - Direct Linear Transform (DLT)
    
    private static func dlt(srcPoints: [CGPoint], dstPoints: [CGPoint]) -> [[Double]]? {
        let n = srcPoints.count
        
        debugLog("=== DLT with \(n) points ===")
        
        // Normalize points
        let (srcNorm, srcT) = normalizePoints(srcPoints)
        let (dstNorm, dstT) = normalizePoints(dstPoints)
        
        debugLog("Source normalization matrix T_src:")
        for row in srcT {
            debugLog("  [\(row.map { String(format: "%10.6f", $0) }.joined(separator: ", "))]")
        }
        debugLog("Dest normalization matrix T_dst:")
        for row in dstT {
            debugLog("  [\(row.map { String(format: "%10.6f", $0) }.joined(separator: ", "))]")
        }
        
        debugLog("Normalized source points:")
        for (i, p) in srcNorm.enumerated() {
            debugLog("  srcNorm[\(i)]: (\(String(format: "%.6f", p.x)), \(String(format: "%.6f", p.y)))")
        }
        debugLog("Normalized dest points:")
        for (i, p) in dstNorm.enumerated() {
            debugLog("  dstNorm[\(i)]: (\(String(format: "%.6f", p.x)), \(String(format: "%.6f", p.y)))")
        }
        
        // Build matrix A in COLUMN-MAJOR order for LAPACK
        // A is (2n x 9) matrix, stored as column-major: A[row + col * numRows]
        let numRows = 2 * n
        let numCols = 9
        var A = [Double](repeating: 0, count: numRows * numCols)
        
        for i in 0..<n {
            let x = Double(srcNorm[i].x), y = Double(srcNorm[i].y)
            let xp = Double(dstNorm[i].x), yp = Double(dstNorm[i].y)
            
            let row1 = 2 * i, row2 = 2 * i + 1
            
            // Row 1: [-x, -y, -1, 0, 0, 0, x*x', y*x', x']
            // Column-major: A[row + col * numRows]
            A[row1 + 0 * numRows] = -x
            A[row1 + 1 * numRows] = -y
            A[row1 + 2 * numRows] = -1
            A[row1 + 3 * numRows] = 0
            A[row1 + 4 * numRows] = 0
            A[row1 + 5 * numRows] = 0
            A[row1 + 6 * numRows] = x * xp
            A[row1 + 7 * numRows] = y * xp
            A[row1 + 8 * numRows] = xp
            
            // Row 2: [0, 0, 0, -x, -y, -1, x*y', y*y', y']
            A[row2 + 0 * numRows] = 0
            A[row2 + 1 * numRows] = 0
            A[row2 + 2 * numRows] = 0
            A[row2 + 3 * numRows] = -x
            A[row2 + 4 * numRows] = -y
            A[row2 + 5 * numRows] = -1
            A[row2 + 6 * numRows] = x * yp
            A[row2 + 7 * numRows] = y * yp
            A[row2 + 8 * numRows] = yp
        }
        
        debugLog("Matrix A (first 4 rows, column-major stored):")
        for i in 0..<min(4, numRows) {
            let row = (0..<numCols).map { String(format: "%8.4f", A[i + $0 * numRows]) }.joined(separator: " ")
            debugLog("  [\(row)]")
        }
        
        guard let h = solveSVD(A: A, rows: numRows, cols: numCols) else {
            debugLog("ERROR: SVD failed")
            return nil
        }
        
        debugLog("SVD solution h (raw):")
        debugLog("  [\(h.map { String(format: "%.8f", $0) }.joined(separator: ", "))]")
        
        var Hnorm = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]]
        
        debugLog("H_normalized (before denormalization):")
        for row in Hnorm {
            debugLog("  [\(row.map { String(format: "%12.8f", $0) }.joined(separator: ", "))]")
        }
        
        // Denormalize: H = dstT^(-1) * Hnorm * srcT
        let dstTinv = invert3x3(dstT)
        debugLog("T_dst^(-1):")
        for row in dstTinv {
            debugLog("  [\(row.map { String(format: "%10.6f", $0) }.joined(separator: ", "))]")
        }
        
        let temp = multiply3x3(dstTinv, Hnorm)
        let H = multiply3x3(temp, srcT)
        
        debugLog("H (before normalization, H[2][2]=\(H[2][2])):")
        for row in H {
            debugLog("  [\(row.map { String(format: "%12.8f", $0) }.joined(separator: ", "))]")
        }
        
        // Normalize so H[2][2] = 1
        let scale = H[2][2]
        guard abs(scale) > 1e-10 else {
            debugLog("ERROR: H[2][2] is too small: \(scale)")
            return nil
        }
        
        let Hfinal = H.map { row in row.map { $0 / scale } }
        
        debugLog("=== FINAL HOMOGRAPHY ===")
        for row in Hfinal {
            debugLog("  [\(row.map { String(format: "%15.8e", $0) }.joined(separator: ", "))]")
        }
        
        // Test: apply H to first source point
        let testPt = srcPoints[0]
        let projPt = applyHomography(Hfinal, to: testPt)
        debugLog("Verification: src[0]=(\(String(format: "%.2f", testPt.x)), \(String(format: "%.2f", testPt.y))) -> (\(String(format: "%.2f", projPt.x)), \(String(format: "%.2f", projPt.y)))")
        debugLog("Expected dst[0]=(\(String(format: "%.2f", dstPoints[0].x)), \(String(format: "%.2f", dstPoints[0].y)))")
        
        return Hfinal
    }
    
    // MARK: - RANSAC
    
    private static func ransac(
        srcPoints: [CGPoint],
        dstPoints: [CGPoint],
        threshold: CGFloat,
        maxIterations: Int = 1000
    ) -> (matrix: [[Double]], inlierMask: [Bool])? {
        
        let n = srcPoints.count
        var bestH: [[Double]]?
        var bestInlierCount = 0
        var bestMask = Array(repeating: false, count: n)
        let thresholdSq = Double(threshold * threshold)
        
        debugLog("=== RANSAC with \(maxIterations) iterations, threshold=\(threshold) ===")
        
        for iter in 0..<maxIterations {
            // Random 4 points
            var indices = Array(0..<n)
            indices.shuffle()
            let sample = Array(indices.prefix(4))
            
            guard let H = dlt(
                srcPoints: sample.map { srcPoints[$0] },
                dstPoints: sample.map { dstPoints[$0] }
            ) else { continue }
            
            // Count inliers
            var mask = Array(repeating: false, count: n)
            var count = 0
            
            for i in 0..<n {
                let proj = applyHomography(H, to: srcPoints[i])
                let dx = Double(proj.x - dstPoints[i].x)
                let dy = Double(proj.y - dstPoints[i].y)
                let distSq = dx*dx + dy*dy
                if distSq < thresholdSq {
                    mask[i] = true
                    count += 1
                }
            }
            
            if count > bestInlierCount {
                bestInlierCount = count
                bestMask = mask
                bestH = H
                if iter < 10 || count == n {
                    debugLog("Iter \(iter): \(count)/\(n) inliers")
                }
            }
        }
        
        debugLog("Best inlier count: \(bestInlierCount)/\(n)")
        
        // Refine with all inliers
        if bestInlierCount >= 4 {
            let inlierSrc = zip(srcPoints, bestMask).filter { $0.1 }.map { $0.0 }
            let inlierDst = zip(dstPoints, bestMask).filter { $0.1 }.map { $0.0 }
            debugLog("Refining with \(inlierSrc.count) inliers...")
            if let refined = dlt(srcPoints: inlierSrc, dstPoints: inlierDst) {
                return (refined, bestMask)
            }
        }
        
        if let H = bestH { return (H, bestMask) }
        return nil
    }
    
    // MARK: - Helper Functions
    
    /// Normalize points for numerical stability
    private static func normalizePoints(_ points: [CGPoint]) -> ([CGPoint], [[Double]]) {
        var cx = 0.0, cy = 0.0
        for p in points { cx += Double(p.x); cy += Double(p.y) }
        cx /= Double(points.count)
        cy /= Double(points.count)
        
        var avgDist = 0.0
        for p in points {
            let dx = Double(p.x) - cx, dy = Double(p.y) - cy
            avgDist += sqrt(dx*dx + dy*dy)
        }
        avgDist /= Double(points.count)
        
        let scale = avgDist > 1e-10 ? sqrt(2.0) / avgDist : 1.0
        
        debugLog("  Centroid: (\(String(format: "%.4f", cx)), \(String(format: "%.4f", cy))), avgDist: \(String(format: "%.4f", avgDist)), scale: \(String(format: "%.6f", scale))")
        
        let T: [[Double]] = [
            [scale, 0, -scale * cx],
            [0, scale, -scale * cy],
            [0, 0, 1]
        ]
        
        let normalized = points.map {
            CGPoint(x: scale * (Double($0.x) - cx), y: scale * (Double($0.y) - cy))
        }
        
        return (normalized, T)
    }
    
    /// Solve Ah = 0 using SVD
    private static func solveSVD(A: [Double], rows: Int, cols: Int) -> [Double]? {
        var a = A
        var m = __CLPK_integer(rows)
        var n = __CLPK_integer(cols)
        var lda = m
        
        var s = [Double](repeating: 0, count: min(rows, cols))
        var u = [Double](repeating: 0, count: rows * rows)
        var ldu = m
        var vt = [Double](repeating: 0, count: cols * cols)
        var ldvt = n
        
        var workSize: Double = 0
        var lwork: __CLPK_integer = -1
        var info: __CLPK_integer = 0
        
        // Query work size
        dgesvd_(
            UnsafeMutablePointer(mutating: ("A" as NSString).utf8String),
            UnsafeMutablePointer(mutating: ("A" as NSString).utf8String),
            &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt,
            &workSize, &lwork, &info
        )
        
        lwork = __CLPK_integer(workSize)
        var work = [Double](repeating: 0, count: Int(lwork))
        
        // Compute SVD
        dgesvd_(
            UnsafeMutablePointer(mutating: ("A" as NSString).utf8String),
            UnsafeMutablePointer(mutating: ("A" as NSString).utf8String),
            &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt,
            &work, &lwork, &info
        )
        
        guard info == 0 else {
            debugLog("SVD failed with info=\(info)")
            return nil
        }
        
        debugLog("SVD singular values: [\(s.map { String(format: "%.6f", $0) }.joined(separator: ", "))]")
        
        // Debug: print V^T matrix
        debugLog("V^T matrix (cols=\(cols)):")
        for row in 0..<cols {
            var rowVals = [Double]()
            for col in 0..<cols {
                // Column-major: element (row, col) at index col * numRows + row
                rowVals.append(vt[col * cols + row])
            }
            debugLog("  row \(row): [\(rowVals.map { String(format: "%10.6f", $0) }.joined(separator: ", "))]")
        }
        
        // Last row of V^T (smallest singular value)
        // LAPACK stores V^T in column-major order (Fortran convention)
        // To get row (cols-1) of V^T, we need element (cols-1, i) for each i
        // In column-major: element (row, col) is at index: col * numRows + row
        var h = [Double](repeating: 0, count: cols)
        for i in 0..<cols {
            h[i] = vt[i * cols + (cols - 1)]  // Column-major access for last row
        }
        
        debugLog("Extracted h (last row of V^T): [\(h.map { String(format: "%.8f", $0) }.joined(separator: ", "))]")
        
        return h
    }
    
    /// Invert 3x3 matrix
    private static func invert3x3(_ m: [[Double]]) -> [[Double]] {
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        
        guard abs(det) > 1e-10 else { return [[1,0,0], [0,1,0], [0,0,1]] }
        let inv = 1.0 / det
        
        return [
            [inv * (m[1][1]*m[2][2] - m[1][2]*m[2][1]),
             inv * (m[0][2]*m[2][1] - m[0][1]*m[2][2]),
             inv * (m[0][1]*m[1][2] - m[0][2]*m[1][1])],
            [inv * (m[1][2]*m[2][0] - m[1][0]*m[2][2]),
             inv * (m[0][0]*m[2][2] - m[0][2]*m[2][0]),
             inv * (m[0][2]*m[1][0] - m[0][0]*m[1][2])],
            [inv * (m[1][0]*m[2][1] - m[1][1]*m[2][0]),
             inv * (m[0][1]*m[2][0] - m[0][0]*m[2][1]),
             inv * (m[0][0]*m[1][1] - m[0][1]*m[1][0])]
        ]
    }
    
    /// Multiply 3x3 matrices
    private static func multiply3x3(_ a: [[Double]], _ b: [[Double]]) -> [[Double]] {
        var r = [[Double]](repeating: [Double](repeating: 0, count: 3), count: 3)
        for i in 0..<3 {
            for j in 0..<3 {
                for k in 0..<3 { r[i][j] += a[i][k] * b[k][j] }
            }
        }
        return r
    }
    
    /// Apply homography to point
    static func applyHomography(_ H: [[Double]], to p: CGPoint) -> CGPoint {
        let x = Double(p.x), y = Double(p.y)
        let w = H[2][0] * x + H[2][1] * y + H[2][2]
        guard abs(w) > 1e-10 else { return p }
        return CGPoint(
            x: (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
            y: (H[1][0] * x + H[1][1] * y + H[1][2]) / w
        )
    }
}
