import SwiftUI
import UniformTypeIdentifiers

// MARK: - Benchmark View

struct BenchmarkView: View {
    @StateObject private var benchmarker = ECGPipelineBenchmarker()
    @State private var loadedImages: [UIImage] = []
    @State private var imageNames: [String] = []
    @State private var report: BenchmarkReport?
    @State private var showingImagePicker = false
    @State private var showingShareSheet = false
    @State private var exportURLs: [URL] = []
    @State private var showAlert = false
    @State private var alertMessage = ""
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Device Info Card
                    DeviceInfoCard(deviceInfo: benchmarker.getDeviceInfo())
                    
                    // Model Sizes Card
                    ModelSizesCard(modelSizes: benchmarker.getModelSizes())
                    
                    // Image Loading Section
                    ImageLoadingSection(
                        loadedCount: loadedImages.count,
                        onLoadFromBundle: loadImagesFromBundle,
                        onLoadFromFolder: { showingImagePicker = true }
                    )
                    
                    // Run Benchmark Section
                    if !loadedImages.isEmpty {
                        RunBenchmarkSection(
                            benchmarker: benchmarker,
                            imageCount: loadedImages.count,
                            onRun: runBenchmark
                        )
                    }
                    
                    // Progress Section
                    if benchmarker.isRunning {
                        ProgressSection(benchmarker: benchmarker)
                    }
                    
                    // Results Section
                    if let report = report {
                        ResultsSection(
                            report: report,
                            onExport: exportReport
                        )
                    }
                }
                .padding()
            }
            .navigationTitle("Pipeline Benchmark")
            .sheet(isPresented: $showingImagePicker) {
                FolderPicker(onSelect: loadImagesFromFolder)
            }
            .sheet(isPresented: $showingShareSheet) {
                if !exportURLs.isEmpty {
                    ShareSheet(activityItems: exportURLs)
                }
            }
            .alert("Benchmark", isPresented: $showAlert) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(alertMessage)
            }
        }
    }
    
    // MARK: - Actions
    
    private func loadImagesFromBundle() {
        loadedImages.removeAll()
        imageNames.removeAll()
        
        let bundle = Bundle.main
        let fileManager = FileManager.default
        
        print("[Benchmark] Bundle path: \(bundle.bundlePath)")
        
        // Method 1: Get ALL image files from bundle root
        let imageExtensions = ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]
        
        for ext in imageExtensions {
            if let urls = bundle.urls(forResourcesWithExtension: ext, subdirectory: nil) {
                print("[Benchmark] Found \(urls.count) .\(ext) files in bundle")
                
                for url in urls {
                    let filename = url.deletingPathExtension().lastPathComponent
                    
                    // Skip non-ECG files (models, code signature, etc.)
                    let skipPrefixes = ["_", "Stage", "ECGClassifier", "AppIcon", "MMECG"]
                    let shouldSkip = skipPrefixes.contains { filename.hasPrefix($0) }
                    
                    if !shouldSkip {
                        if let image = UIImage(contentsOfFile: url.path) {
                            loadedImages.append(image)
                            imageNames.append(filename)
                            print("[Benchmark] Loaded: \(url.lastPathComponent)")
                        }
                    }
                }
            }
        }
        
        // Method 2: Also check TestImages folder if exists
        if loadedImages.isEmpty {
            if let testImagesURL = bundle.url(forResource: "TestImages", withExtension: nil) {
                print("[Benchmark] Found TestImages folder")
                loadImagesFromURL(testImagesURL)
            }
        }
        
        // Sort by name
        if !loadedImages.isEmpty {
            let combined = zip(imageNames, loadedImages).sorted { $0.0.localizedStandardCompare($1.0) == .orderedAscending }
            imageNames = combined.map { $0.0 }
            loadedImages = combined.map { $0.1 }
            
            alertMessage = "Loaded \(loadedImages.count) images."
            showAlert = true
        } else {
            alertMessage = "No test images found in bundle."
            showAlert = true
        }
        
        print("[Benchmark] Total loaded: \(loadedImages.count)")
    }
    
    private func loadImagesFromURL(_ url: URL) {
        let fileManager = FileManager.default
        let imageExtensions = ["jpg", "jpeg", "png", "heic", "heif"]
        
        guard let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey]) else {
            return
        }
        
        for case let fileURL as URL in enumerator {
            let ext = fileURL.pathExtension.lowercased()
            if imageExtensions.contains(ext) {
                if let image = UIImage(contentsOfFile: fileURL.path) {
                    loadedImages.append(image)
                    imageNames.append(fileURL.deletingPathExtension().lastPathComponent)
                }
            }
        }
    }
    
    private func loadImagesFromFolder(_ url: URL) {
        loadedImages.removeAll()
        imageNames.removeAll()
        
        let fileManager = FileManager.default
        
        guard let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey]) else {
            alertMessage = "Could not access folder."
            showAlert = true
            return
        }
        
        let imageExtensions = ["jpg", "jpeg", "png", "heic", "heif"]
        
        for case let fileURL as URL in enumerator {
            let ext = fileURL.pathExtension.lowercased()
            if imageExtensions.contains(ext) {
                if let image = UIImage(contentsOfFile: fileURL.path) {
                    loadedImages.append(image)
                    imageNames.append(fileURL.deletingPathExtension().lastPathComponent)
                }
            }
        }
        
        // Sort by name
        let combined = zip(imageNames, loadedImages).sorted { $0.0 < $1.0 }
        imageNames = combined.map { $0.0 }
        loadedImages = combined.map { $0.1 }
        
        alertMessage = "Loaded \(loadedImages.count) images."
        showAlert = true
    }
    
    private func runBenchmark() {
        guard !loadedImages.isEmpty else { return }
        
        benchmarker.runBenchmark(images: loadedImages, imageNames: imageNames) { result in
            self.report = result
        }
    }
    
    private func exportReport() {
        guard let report = report else { return }
        
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let reportDir = documentsURL.appendingPathComponent("BenchmarkReports/\(report.reportId)")
        
        do {
            exportURLs = try benchmarker.saveReport(report, to: reportDir)
            showingShareSheet = true
        } catch {
            alertMessage = "Failed to export: \(error.localizedDescription)"
            showAlert = true
        }
    }
}

// MARK: - Device Info Card

struct DeviceInfoCard: View {
    let deviceInfo: DeviceInfo
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Device Information", systemImage: "iphone")
                .font(.headline)
            
            Divider()
            
            InfoRow(label: "Model", value: deviceInfo.modelIdentifier)
            InfoRow(label: "iOS Version", value: deviceInfo.systemVersion)
            InfoRow(label: "Processors", value: "\(deviceInfo.processorCount) cores")
            InfoRow(label: "Memory", value: String(format: "%.1f GB", deviceInfo.physicalMemoryGB))
            InfoRow(label: "Thermal State", value: deviceInfo.thermalState)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Model Sizes Card

struct ModelSizesCard: View {
    let modelSizes: ModelSizeInfo
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Model Sizes", systemImage: "doc.zipper")
                .font(.headline)
            
            Divider()
            
            InfoRow(label: "Stage 0", value: formatSize(modelSizes.stage0SizeKB))
            InfoRow(label: "Stage 1", value: formatSize(modelSizes.stage1SizeKB))
            InfoRow(label: "Stage 2", value: formatSize(modelSizes.stage2SizeKB))
            InfoRow(label: "Classifier", value: formatSize(modelSizes.classifierSizeKB))
            
            Divider()
            
            InfoRow(label: "Total Models", value: formatSize(modelSizes.totalModelSizeKB))
            if let appSize = modelSizes.appSizeKB {
                InfoRow(label: "App Bundle", value: formatSize(appSize))
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func formatSize(_ kb: Int64) -> String {
        if kb >= 1024 {
            return String(format: "%.2f MB", Double(kb) / 1024)
        }
        return "\(kb) KB"
    }
}

// MARK: - Image Loading Section

struct ImageLoadingSection: View {
    let loadedCount: Int
    let onLoadFromBundle: () -> Void
    let onLoadFromFolder: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Test Images", systemImage: "photo.stack")
                .font(.headline)
            
            Divider()
            
            HStack {
                Text("Loaded:")
                Spacer()
                Text("\(loadedCount) images")
                    .fontWeight(.semibold)
                    .foregroundColor(loadedCount > 0 ? .green : .secondary)
            }
            
            HStack(spacing: 12) {
                Button(action: onLoadFromBundle) {
                    Label("From Bundle", systemImage: "app.badge")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                
                Button(action: onLoadFromFolder) {
                    Label("From Folder", systemImage: "folder")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Run Benchmark Section

struct RunBenchmarkSection: View {
    @ObservedObject var benchmarker: ECGPipelineBenchmarker
    let imageCount: Int
    let onRun: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Run Benchmark", systemImage: "play.circle")
                .font(.headline)
            
            Divider()
            
            Text("Will process \(imageCount) images and measure latency, memory, and CPU usage for each pipeline stage.")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Button(action: onRun) {
                HStack {
                    Image(systemName: "bolt.fill")
                    Text("Start Benchmark")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(benchmarker.isRunning ? Color.gray : Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .disabled(benchmarker.isRunning)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Progress Section

struct ProgressSection: View {
    @ObservedObject var benchmarker: ECGPipelineBenchmarker
    @State private var startTime: Date?
    
    var body: some View {
        VStack(spacing: 16) {
            // Circular progress
            ZStack {
                // Background circle
                Circle()
                    .stroke(Color.gray.opacity(0.2), lineWidth: 12)
                    .frame(width: 120, height: 120)
                
                // Progress circle
                Circle()
                    .trim(from: 0, to: CGFloat(benchmarker.currentProgress))
                    .stroke(
                        LinearGradient(
                            colors: [.blue, .cyan],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        style: StrokeStyle(lineWidth: 12, lineCap: .round)
                    )
                    .frame(width: 120, height: 120)
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 0.3), value: benchmarker.currentProgress)
                
                // Percentage text
                VStack(spacing: 2) {
                    Text("\(Int(benchmarker.currentProgress * 100))%")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(.primary)
                    
                    Text("\(benchmarker.processedCount)/\(benchmarker.totalCount)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.vertical, 8)
            
            // Status message
            Text(benchmarker.currentStatus)
                .font(.subheadline)
                .foregroundColor(.primary)
                .multilineTextAlignment(.center)
                .lineLimit(2)
            
            // Linear progress bar
            VStack(spacing: 4) {
                ProgressView(value: benchmarker.currentProgress)
                    .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                    .scaleEffect(y: 2)
                
                HStack {
                    Text("Processed: \(benchmarker.processedCount)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text("Remaining: \(benchmarker.totalCount - benchmarker.processedCount)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.top, 8)
            
            // Estimated time
            if benchmarker.processedCount > 0 {
                EstimatedTimeView(
                    processedCount: benchmarker.processedCount,
                    totalCount: benchmarker.totalCount,
                    startTime: startTime ?? Date()
                )
            }
            
            // Pulsing indicator
            HStack(spacing: 8) {
                PulsingDot()
                Text("Benchmark in progress...")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.top, 4)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .onAppear {
            startTime = Date()
        }
    }
}

struct EstimatedTimeView: View {
    let processedCount: Int
    let totalCount: Int
    let startTime: Date
    
    var estimatedTimeRemaining: String {
        guard processedCount > 0 else { return "Calculating..." }
        
        let elapsed = Date().timeIntervalSince(startTime)
        let avgTimePerImage = elapsed / Double(processedCount)
        let remaining = avgTimePerImage * Double(totalCount - processedCount)
        
        if remaining < 60 {
            return "\(Int(remaining))s remaining"
        } else if remaining < 3600 {
            let minutes = Int(remaining) / 60
            let seconds = Int(remaining) % 60
            return "\(minutes)m \(seconds)s remaining"
        } else {
            let hours = Int(remaining) / 3600
            let minutes = (Int(remaining) % 3600) / 60
            return "\(hours)h \(minutes)m remaining"
        }
    }
    
    var avgTimePerImage: String {
        guard processedCount > 0 else { return "-" }
        let elapsed = Date().timeIntervalSince(startTime)
        let avg = elapsed / Double(processedCount)
        return String(format: "%.1fs/image", avg)
    }
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("⏱ \(estimatedTimeRemaining)")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.blue)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 2) {
                Text("Avg: \(avgTimePerImage)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Color.blue.opacity(0.1))
        .cornerRadius(8)
    }
}

struct PulsingDot: View {
    @State private var isPulsing = false
    
    var body: some View {
        Circle()
            .fill(Color.green)
            .frame(width: 8, height: 8)
            .scaleEffect(isPulsing ? 1.2 : 0.8)
            .opacity(isPulsing ? 1.0 : 0.5)
            .animation(
                Animation.easeInOut(duration: 0.8)
                    .repeatForever(autoreverses: true),
                value: isPulsing
            )
            .onAppear {
                isPulsing = true
            }
    }
}

// MARK: - Results Section

struct ResultsSection: View {
    let report: BenchmarkReport
    let onExport: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Results", systemImage: "chart.bar")
                .font(.headline)
            
            Divider()
            
            // Summary stats
            Group {
                Text("Latency Summary (ms)")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                
                LatencyRow(label: "Stage 0", stats: report.latencySummary.stage0)
                LatencyRow(label: "Stage 1", stats: report.latencySummary.stage1)
                LatencyRow(label: "Stage 2", stats: report.latencySummary.stage2)
                LatencyRow(label: "Classification", stats: report.latencySummary.classification)
                LatencyRow(label: "Total Digitization", stats: report.latencySummary.totalDigitization)
                LatencyRow(label: "End-to-End", stats: report.latencySummary.endToEnd)
            }
            
            Divider()
            
            Group {
                Text("Memory Summary (MB)")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                
                LatencyRow(label: "Peak", stats: report.memorySummary.peakMemoryMB)
                LatencyRow(label: "Average", stats: report.memorySummary.avgMemoryMB)
            }
            
            Divider()
            
            Group {
                Text("CPU Summary (%)")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                
                LatencyRow(label: "Average", stats: report.cpuSummary.avgCpuPercent)
            }
            
            Divider()
            
            HStack {
                Text("Success Rate:")
                Spacer()
                Text("\(report.successfulProcessed)/\(report.totalImagesProcessed)")
                    .fontWeight(.semibold)
                    .foregroundColor(report.failedProcessed == 0 ? .green : .orange)
            }
            
            Button(action: onExport) {
                HStack {
                    Image(systemName: "square.and.arrow.up")
                    Text("Export Report")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.green)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct LatencyRow: View {
    let label: String
    let stats: StatsSummary
    
    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
            Spacer()
            Text(String(format: "%.1f", stats.mean))
                .font(.caption)
                .fontWeight(.semibold)
            Text("±\(String(format: "%.1f", stats.stdDev))")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Helper Views

struct InfoRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
        }
        .font(.subheadline)
    }
}

// MARK: - Share Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - Folder Picker

struct FolderPicker: UIViewControllerRepresentable {
    let onSelect: (URL) -> Void
    
    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.folder])
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(onSelect: onSelect)
    }
    
    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onSelect: (URL) -> Void
        
        init(onSelect: @escaping (URL) -> Void) {
            self.onSelect = onSelect
        }
        
        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }
            
            // Start accessing security-scoped resource
            guard url.startAccessingSecurityScopedResource() else { return }
            defer { url.stopAccessingSecurityScopedResource() }
            
            onSelect(url)
        }
    }
}

// MARK: - Preview

#Preview {
    BenchmarkView()
}
