import Foundation
import CoreML
import UIKit
import os.signpost

// MARK: - Benchmark Data Structures

struct StageBenchmark: Codable {
    let stageName: String
    let latencyMs: Double
    let peakMemoryMB: Double
    let avgMemoryMB: Double
    let cpuUsagePercent: Double
    let gpuUsagePercent: Double?  // May not be available on all devices
}

struct ImageBenchmarkResult: Codable {
    let imageIndex: Int
    let imageName: String
    let timestamp: String
    
    // Stage-level metrics
    let stage0: StageBenchmark
    let stage1: StageBenchmark
    let stage2: StageBenchmark
    let classification: StageBenchmark
    
    // Aggregate metrics
    let totalDigitizationMs: Double
    let totalClassificationMs: Double
    let endToEndMs: Double
    
    let peakMemoryMB: Double
    let avgMemoryMB: Double
    let overallCpuPercent: Double
    
    let success: Bool
    let errorMessage: String?
}

struct DeviceInfo: Codable {
    let modelName: String
    let modelIdentifier: String
    let systemVersion: String
    let processorCount: Int
    let physicalMemoryGB: Double
    let thermalState: String
}

struct ModelSizeInfo: Codable {
    let stage0SizeKB: Int64
    let stage1SizeKB: Int64
    let stage2SizeKB: Int64
    let classifierSizeKB: Int64
    let totalModelSizeKB: Int64
    let appSizeKB: Int64?
}

struct BenchmarkReport: Codable {
    let reportId: String
    let generatedAt: String
    let deviceInfo: DeviceInfo
    let modelSizes: ModelSizeInfo
    let totalImagesProcessed: Int
    let successfulProcessed: Int
    let failedProcessed: Int
    
    // Summary statistics
    let latencySummary: LatencySummary
    let memorySummary: MemorySummary
    let cpuSummary: CPUSummary
    
    // Individual results
    let results: [ImageBenchmarkResult]
}

struct LatencySummary: Codable {
    let stage0: StatsSummary
    let stage1: StatsSummary
    let stage2: StatsSummary
    let classification: StatsSummary
    let totalDigitization: StatsSummary
    let endToEnd: StatsSummary
}

struct MemorySummary: Codable {
    let peakMemoryMB: StatsSummary
    let avgMemoryMB: StatsSummary
}

struct CPUSummary: Codable {
    let avgCpuPercent: StatsSummary
}

struct StatsSummary: Codable {
    let min: Double
    let max: Double
    let mean: Double
    let median: Double
    let stdDev: Double
    let p95: Double
    let p99: Double
}

// MARK: - Memory & CPU Tracking

class ResourceMonitor {
    private var memoryReadings: [Double] = []
    private var cpuReadings: [Double] = []
    private var monitorTimer: Timer?
    private var isMonitoring = false
    private let sampleIntervalMs: Double = 50  // Sample every 50ms
    
    func startMonitoring() {
        memoryReadings.removeAll()
        cpuReadings.removeAll()
        isMonitoring = true
        
        // Use a high-frequency timer for sampling
        monitorTimer = Timer.scheduledTimer(withTimeInterval: sampleIntervalMs / 1000.0, repeats: true) { [weak self] _ in
            guard let self = self, self.isMonitoring else { return }
            self.memoryReadings.append(self.getCurrentMemoryMB())
            self.cpuReadings.append(self.getCPUUsage())
        }
        RunLoop.current.add(monitorTimer!, forMode: .common)
    }
    
    func stopMonitoring() -> (peakMemory: Double, avgMemory: Double, avgCpu: Double) {
        isMonitoring = false
        monitorTimer?.invalidate()
        monitorTimer = nil
        
        let peakMem = memoryReadings.max() ?? 0
        let avgMem = memoryReadings.isEmpty ? 0 : memoryReadings.reduce(0, +) / Double(memoryReadings.count)
        let avgCpu = cpuReadings.isEmpty ? 0 : cpuReadings.reduce(0, +) / Double(cpuReadings.count)
        
        return (peakMem, avgMem, avgCpu)
    }
    
    func getCurrentMemoryMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return Double(info.resident_size) / (1024 * 1024)  // Convert to MB
        }
        return 0
    }
    
    func getCPUUsage() -> Double {
        var threads: thread_act_array_t?
        var threadCount: mach_msg_type_number_t = 0
        
        guard task_threads(mach_task_self_, &threads, &threadCount) == KERN_SUCCESS,
              let threadList = threads else {
            return 0
        }
        
        var totalCPU: Double = 0
        let threadInfoCount = MemoryLayout<thread_basic_info_data_t>.size / MemoryLayout<integer_t>.size
        
        for i in 0..<Int(threadCount) {
            var info = thread_basic_info_data_t()
            var infoCount = mach_msg_type_number_t(threadInfoCount)
            
            let result = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: threadInfoCount) {
                    thread_info(threadList[i], thread_flavor_t(THREAD_BASIC_INFO), $0, &infoCount)
                }
            }
            
            if result == KERN_SUCCESS && (info.flags & TH_FLAGS_IDLE) == 0 {
                totalCPU += Double(info.cpu_usage) / Double(TH_USAGE_SCALE) * 100.0
            }
        }
        
        // Deallocate thread list
        let size = vm_size_t(MemoryLayout<thread_t>.stride * Int(threadCount))
        vm_deallocate(mach_task_self_, vm_address_t(bitPattern: threadList), size)
        
        return totalCPU
    }
}

// MARK: - Pipeline Benchmarker

class ECGPipelineBenchmarker: ObservableObject {
    
    @Published var isRunning = false
    @Published var currentProgress: Float = 0
    @Published var currentStatus: String = ""
    @Published var processedCount: Int = 0
    @Published var totalCount: Int = 0
    
    private var pipeline: ECGPipeline?
    private var classifier: ECGClassifierProcessor?
    private let resourceMonitor = ResourceMonitor()
    
    private var results: [ImageBenchmarkResult] = []
    
    // For stage-level timing
    var stageStartTime: CFAbsoluteTime = 0
    var stageTimes: [String: Double] = [:]
    var stageMemory: [String: (peak: Double, avg: Double)] = [:]
    var stageCPU: [String: Double] = [:]
    
    init() {
        setupPipeline()
    }
    
    private func setupPipeline() {
        do {
            pipeline = try ECGPipeline()
            classifier = try ECGClassifierProcessor()
        } catch {
            print("[Benchmark] Failed to setup pipeline: \(error)")
        }
    }
    
    // MARK: - Device Info
    
    func getDeviceInfo() -> DeviceInfo {
        var systemInfo = utsname()
        uname(&systemInfo)
        let modelIdentifier = withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0) ?? "Unknown"
            }
        }
        
        let thermalState: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermalState = "nominal"
        case .fair: thermalState = "fair"
        case .serious: thermalState = "serious"
        case .critical: thermalState = "critical"
        @unknown default: thermalState = "unknown"
        }
        
        return DeviceInfo(
            modelName: UIDevice.current.model,
            modelIdentifier: modelIdentifier,
            systemVersion: UIDevice.current.systemVersion,
            processorCount: ProcessInfo.processInfo.processorCount,
            physicalMemoryGB: Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024),
            thermalState: thermalState
        )
    }
    
    // MARK: - Model Sizes
    
    func getModelSizes() -> ModelSizeInfo {
        let bundle = Bundle.main
        
        func getModelSize(_ name: String) -> Int64 {
            guard let url = bundle.url(forResource: name, withExtension: "mlmodelc") else {
                return 0
            }
            return folderSize(at: url)
        }
        
        let stage0 = getModelSize("Stage0")
        let stage1 = getModelSize("Stage1")
        let stage2 = getModelSize("Stage2")
        let classifier = getModelSize("ECGClassifier")
        
        // Get app bundle size
        let appSize = folderSize(at: bundle.bundleURL)
        
        return ModelSizeInfo(
            stage0SizeKB: stage0 / 1024,
            stage1SizeKB: stage1 / 1024,
            stage2SizeKB: stage2 / 1024,
            classifierSizeKB: classifier / 1024,
            totalModelSizeKB: (stage0 + stage1 + stage2 + classifier) / 1024,
            appSizeKB: appSize / 1024
        )
    }
    
    private func folderSize(at url: URL) -> Int64 {
        let fileManager = FileManager.default
        var totalSize: Int64 = 0
        
        guard let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }
        
        for case let fileURL as URL in enumerator {
            guard let attributes = try? fileManager.attributesOfItem(atPath: fileURL.path),
                  let fileSize = attributes[.size] as? Int64 else {
                continue
            }
            totalSize += fileSize
        }
        
        return totalSize
    }
    
    // MARK: - Run Benchmark
    
    func runBenchmark(images: [UIImage], imageNames: [String], completion: @escaping (BenchmarkReport) -> Void) {
        guard let pipeline = pipeline else {
            print("[Benchmark] Pipeline not initialized")
            return
        }
        
        DispatchQueue.main.async {
            self.isRunning = true
            self.processedCount = 0
            self.totalCount = images.count
            self.results.removeAll()
        }
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            for (index, image) in images.enumerated() {
                let imageName = index < imageNames.count ? imageNames[index] : "image_\(index)"
                
                DispatchQueue.main.async {
                    self.currentStatus = "Processing \(imageName)..."
                    self.currentProgress = Float(index) / Float(images.count)
                }
                
                let result = self.benchmarkSingleImage(
                    image: image,
                    imageName: imageName,
                    index: index,
                    pipeline: pipeline
                )
                
                self.results.append(result)
                
                DispatchQueue.main.async {
                    self.processedCount = index + 1
                }
                
                // Small delay between images to let system stabilize
                Thread.sleep(forTimeInterval: 0.5)
            }
            
            // Generate report
            let report = self.generateReport()
            
            DispatchQueue.main.async {
                self.isRunning = false
                self.currentStatus = "Complete!"
                self.currentProgress = 1.0
                completion(report)
            }
        }
    }
    
    private func benchmarkSingleImage(image: UIImage, imageName: String, index: Int, pipeline: ECGPipeline) -> ImageBenchmarkResult {
        
        let dateFormatter = ISO8601DateFormatter()
        let timestamp = dateFormatter.string(from: Date())
        
        // Reset stage tracking
        stageTimes.removeAll()
        stageMemory.removeAll()
        stageCPU.removeAll()
        
        var overallSuccess = true
        var errorMessage: String?
        var signals: [String: ECGSignal] = [:]
        
        // Overall timing and monitoring
        let overallStartTime = CFAbsoluteTimeGetCurrent()
        let overallMonitor = ResourceMonitor()
        overallMonitor.startMonitoring()
        
        // ===== STAGE 0: Keypoint Detection =====
        let stage0Result = benchmarkStage(name: "Stage0") {
            // This would call the actual stage0 processor
            // For now, we'll measure through the pipeline
        }
        
        // ===== Run full pipeline with timing hooks =====
        let digitizationStart = CFAbsoluteTimeGetCurrent()
        
        let semaphore = DispatchSemaphore(value: 0)
        var pipelineResult: ECGPipelineResult?
        
        // Stage timing will be captured via progress handler
        var lastStageTime = digitizationStart
        var currentStageName = "Stage0"
        let stageMonitor = ResourceMonitor()
        stageMonitor.startMonitoring()
        
        pipeline.processAsync(
            image: image,
            targetSignalLength: 5000,
            progressHandler: { [weak self] progress, message in
                guard let self = self else { return }
                
                let now = CFAbsoluteTimeGetCurrent()
                
                // Detect stage transitions based on progress/message
                var newStageName: String?
                if message.contains("keypoint") || message.contains("Step 1") || progress < 0.25 {
                    newStageName = "Stage0"
                } else if message.contains("grid") || message.contains("Step 2") || progress < 0.5 {
                    newStageName = "Stage1"
                } else if message.contains("signal") || message.contains("Step 3") || message.contains("extract") {
                    newStageName = "Stage2"
                }
                
                if let newStage = newStageName, newStage != currentStageName {
                    // Record previous stage
                    let stageTime = (now - lastStageTime) * 1000  // Convert to ms
                    self.stageTimes[currentStageName] = stageTime
                    
                    let (peak, avg, cpu) = stageMonitor.stopMonitoring()
                    self.stageMemory[currentStageName] = (peak, avg)
                    self.stageCPU[currentStageName] = cpu
                    
                    // Start new stage monitoring
                    stageMonitor.startMonitoring()
                    lastStageTime = now
                    currentStageName = newStage
                }
            },
            completion: { result in
                pipelineResult = result
                semaphore.signal()
            }
        )
        
        semaphore.wait()
        
        // Record final stage
        let digitizationEnd = CFAbsoluteTimeGetCurrent()
        let finalStageTime = (digitizationEnd - lastStageTime) * 1000
        stageTimes[currentStageName] = finalStageTime
        let (peak, avg, cpu) = stageMonitor.stopMonitoring()
        stageMemory[currentStageName] = (peak, avg)
        stageCPU[currentStageName] = cpu
        
        let totalDigitizationMs = (digitizationEnd - digitizationStart) * 1000
        
        if let result = pipelineResult, result.success {
            signals = result.signals
        } else {
            overallSuccess = false
            errorMessage = pipelineResult?.errorMessage ?? "Digitization failed"
        }
        
        // ===== CLASSIFICATION =====
        var classificationMs: Double = 0
        var classificationMemory: (peak: Double, avg: Double) = (0, 0)
        var classificationCPU: Double = 0
        
        if overallSuccess, let classifier = classifier, classifier.isModelLoaded {
            let classMonitor = ResourceMonitor()
            classMonitor.startMonitoring()
            
            let classStart = CFAbsoluteTimeGetCurrent()
            
            do {
                let _ = try classifier.classify(signals: signals)
            } catch {
                // Classification failed but digitization succeeded
                print("[Benchmark] Classification failed: \(error)")
            }
            
            let classEnd = CFAbsoluteTimeGetCurrent()
            classificationMs = (classEnd - classStart) * 1000
            
            let (classPeak, classAvg, classCpu) = classMonitor.stopMonitoring()
            classificationMemory = (classPeak, classAvg)
            classificationCPU = classCpu
        }
        
        // Overall metrics
        let overallEndTime = CFAbsoluteTimeGetCurrent()
        let endToEndMs = (overallEndTime - overallStartTime) * 1000
        let (overallPeak, overallAvg, overallCpu) = overallMonitor.stopMonitoring()
        
        // Build stage benchmarks
        let stage0Benchmark = StageBenchmark(
            stageName: "Stage0",
            latencyMs: stageTimes["Stage0"] ?? 0,
            peakMemoryMB: stageMemory["Stage0"]?.peak ?? 0,
            avgMemoryMB: stageMemory["Stage0"]?.avg ?? 0,
            cpuUsagePercent: stageCPU["Stage0"] ?? 0,
            gpuUsagePercent: nil
        )
        
        let stage1Benchmark = StageBenchmark(
            stageName: "Stage1",
            latencyMs: stageTimes["Stage1"] ?? 0,
            peakMemoryMB: stageMemory["Stage1"]?.peak ?? 0,
            avgMemoryMB: stageMemory["Stage1"]?.avg ?? 0,
            cpuUsagePercent: stageCPU["Stage1"] ?? 0,
            gpuUsagePercent: nil
        )
        
        let stage2Benchmark = StageBenchmark(
            stageName: "Stage2",
            latencyMs: stageTimes["Stage2"] ?? 0,
            peakMemoryMB: stageMemory["Stage2"]?.peak ?? 0,
            avgMemoryMB: stageMemory["Stage2"]?.avg ?? 0,
            cpuUsagePercent: stageCPU["Stage2"] ?? 0,
            gpuUsagePercent: nil
        )
        
        let classificationBenchmark = StageBenchmark(
            stageName: "Classification",
            latencyMs: classificationMs,
            peakMemoryMB: classificationMemory.peak,
            avgMemoryMB: classificationMemory.avg,
            cpuUsagePercent: classificationCPU,
            gpuUsagePercent: nil
        )
        
        return ImageBenchmarkResult(
            imageIndex: index,
            imageName: imageName,
            timestamp: timestamp,
            stage0: stage0Benchmark,
            stage1: stage1Benchmark,
            stage2: stage2Benchmark,
            classification: classificationBenchmark,
            totalDigitizationMs: totalDigitizationMs,
            totalClassificationMs: classificationMs,
            endToEndMs: endToEndMs,
            peakMemoryMB: overallPeak,
            avgMemoryMB: overallAvg,
            overallCpuPercent: overallCpu,
            success: overallSuccess,
            errorMessage: errorMessage
        )
    }
    
    private func benchmarkStage(name: String, operation: () -> Void) -> (latency: Double, peakMem: Double, avgMem: Double, cpu: Double) {
        let monitor = ResourceMonitor()
        monitor.startMonitoring()
        
        let start = CFAbsoluteTimeGetCurrent()
        operation()
        let end = CFAbsoluteTimeGetCurrent()
        
        let (peak, avg, cpu) = monitor.stopMonitoring()
        
        return ((end - start) * 1000, peak, avg, cpu)
    }
    
    // MARK: - Generate Report
    
    private func generateReport() -> BenchmarkReport {
        let deviceInfo = getDeviceInfo()
        let modelSizes = getModelSizes()
        
        let successCount = results.filter { $0.success }.count
        let failCount = results.count - successCount
        
        // Calculate statistics
        let latencySummary = calculateLatencySummary()
        let memorySummary = calculateMemorySummary()
        let cpuSummary = calculateCPUSummary()
        
        let dateFormatter = ISO8601DateFormatter()
        
        return BenchmarkReport(
            reportId: UUID().uuidString,
            generatedAt: dateFormatter.string(from: Date()),
            deviceInfo: deviceInfo,
            modelSizes: modelSizes,
            totalImagesProcessed: results.count,
            successfulProcessed: successCount,
            failedProcessed: failCount,
            latencySummary: latencySummary,
            memorySummary: memorySummary,
            cpuSummary: cpuSummary,
            results: results
        )
    }
    
    private func calculateLatencySummary() -> LatencySummary {
        let successResults = results.filter { $0.success }
        
        return LatencySummary(
            stage0: calculateStats(successResults.map { $0.stage0.latencyMs }),
            stage1: calculateStats(successResults.map { $0.stage1.latencyMs }),
            stage2: calculateStats(successResults.map { $0.stage2.latencyMs }),
            classification: calculateStats(successResults.map { $0.classification.latencyMs }),
            totalDigitization: calculateStats(successResults.map { $0.totalDigitizationMs }),
            endToEnd: calculateStats(successResults.map { $0.endToEndMs })
        )
    }
    
    private func calculateMemorySummary() -> MemorySummary {
        let successResults = results.filter { $0.success }
        
        return MemorySummary(
            peakMemoryMB: calculateStats(successResults.map { $0.peakMemoryMB }),
            avgMemoryMB: calculateStats(successResults.map { $0.avgMemoryMB })
        )
    }
    
    private func calculateCPUSummary() -> CPUSummary {
        let successResults = results.filter { $0.success }
        
        return CPUSummary(
            avgCpuPercent: calculateStats(successResults.map { $0.overallCpuPercent })
        )
    }
    
    private func calculateStats(_ values: [Double]) -> StatsSummary {
        guard !values.isEmpty else {
            return StatsSummary(min: 0, max: 0, mean: 0, median: 0, stdDev: 0, p95: 0, p99: 0)
        }
        
        let sorted = values.sorted()
        let count = Double(sorted.count)
        
        let minVal = sorted.first!
        let maxVal = sorted.last!
        let mean = sorted.reduce(0, +) / count
        
        let median: Double
        if sorted.count % 2 == 0 {
            median = (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
        } else {
            median = sorted[sorted.count / 2]
        }
        
        let variance = sorted.map { pow($0 - mean, 2) }.reduce(0, +) / count
        let stdDev = sqrt(variance)
        
        let p95Index = Int(Double(sorted.count) * 0.95)
        let p99Index = Int(Double(sorted.count) * 0.99)
        let p95 = sorted[Swift.min(p95Index, sorted.count - 1)]
        let p99 = sorted[Swift.min(p99Index, sorted.count - 1)]
        
        return StatsSummary(
            min: minVal,
            max: maxVal,
            mean: mean,
            median: median,
            stdDev: stdDev,
            p95: p95,
            p99: p99
        )
    }
    
    // MARK: - Export Functions
    
    func exportToJSON(_ report: BenchmarkReport) -> Data? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try? encoder.encode(report)
    }
    
    func exportToCSV(_ report: BenchmarkReport) -> String {
        var csv = "Image Index,Image Name,Timestamp,Success,"
        csv += "Stage0 Latency (ms),Stage1 Latency (ms),Stage2 Latency (ms),Classification Latency (ms),"
        csv += "Total Digitization (ms),End-to-End (ms),"
        csv += "Peak Memory (MB),Avg Memory (MB),CPU Usage (%)\n"
        
        for result in report.results {
            csv += "\(result.imageIndex),\(result.imageName),\(result.timestamp),\(result.success),"
            csv += "\(String(format: "%.2f", result.stage0.latencyMs)),"
            csv += "\(String(format: "%.2f", result.stage1.latencyMs)),"
            csv += "\(String(format: "%.2f", result.stage2.latencyMs)),"
            csv += "\(String(format: "%.2f", result.classification.latencyMs)),"
            csv += "\(String(format: "%.2f", result.totalDigitizationMs)),"
            csv += "\(String(format: "%.2f", result.endToEndMs)),"
            csv += "\(String(format: "%.2f", result.peakMemoryMB)),"
            csv += "\(String(format: "%.2f", result.avgMemoryMB)),"
            csv += "\(String(format: "%.2f", result.overallCpuPercent))\n"
        }
        
        return csv
    }
    
    func exportSummaryCSV(_ report: BenchmarkReport) -> String {
        var csv = "Metric,Min,Max,Mean,Median,StdDev,P95,P99\n"
        
        func addRow(_ name: String, _ stats: StatsSummary) {
            csv += "\(name),"
            csv += "\(String(format: "%.2f", stats.min)),"
            csv += "\(String(format: "%.2f", stats.max)),"
            csv += "\(String(format: "%.2f", stats.mean)),"
            csv += "\(String(format: "%.2f", stats.median)),"
            csv += "\(String(format: "%.2f", stats.stdDev)),"
            csv += "\(String(format: "%.2f", stats.p95)),"
            csv += "\(String(format: "%.2f", stats.p99))\n"
        }
        
        addRow("Stage0 Latency (ms)", report.latencySummary.stage0)
        addRow("Stage1 Latency (ms)", report.latencySummary.stage1)
        addRow("Stage2 Latency (ms)", report.latencySummary.stage2)
        addRow("Classification Latency (ms)", report.latencySummary.classification)
        addRow("Total Digitization (ms)", report.latencySummary.totalDigitization)
        addRow("End-to-End (ms)", report.latencySummary.endToEnd)
        addRow("Peak Memory (MB)", report.memorySummary.peakMemoryMB)
        addRow("Avg Memory (MB)", report.memorySummary.avgMemoryMB)
        addRow("CPU Usage (%)", report.cpuSummary.avgCpuPercent)
        
        return csv
    }
    
    func saveReport(_ report: BenchmarkReport, to directory: URL) throws -> [URL] {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        
        var savedFiles: [URL] = []
        
        // Save JSON
        if let jsonData = exportToJSON(report) {
            let jsonURL = directory.appendingPathComponent("benchmark_report.json")
            try jsonData.write(to: jsonURL)
            savedFiles.append(jsonURL)
        }
        
        // Save detailed CSV
        let detailCSV = exportToCSV(report)
        let detailURL = directory.appendingPathComponent("benchmark_detailed.csv")
        try detailCSV.write(to: detailURL, atomically: true, encoding: .utf8)
        savedFiles.append(detailURL)
        
        // Save summary CSV
        let summaryCSV = exportSummaryCSV(report)
        let summaryURL = directory.appendingPathComponent("benchmark_summary.csv")
        try summaryCSV.write(to: summaryURL, atomically: true, encoding: .utf8)
        savedFiles.append(summaryURL)
        
        // Save device info
        let deviceInfoURL = directory.appendingPathComponent("device_info.json")
        let deviceData = try JSONEncoder().encode(report.deviceInfo)
        try deviceData.write(to: deviceInfoURL)
        savedFiles.append(deviceInfoURL)
        
        // Save model sizes
        let modelSizesURL = directory.appendingPathComponent("model_sizes.json")
        let modelData = try JSONEncoder().encode(report.modelSizes)
        try modelData.write(to: modelSizesURL)
        savedFiles.append(modelSizesURL)
        
        return savedFiles
    }
}
