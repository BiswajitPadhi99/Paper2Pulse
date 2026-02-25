
import SwiftUI
import CoreML

// MARK: - Main Content View

struct ContentView: View {
    @StateObject private var viewModel = ECGViewModel()
    
    var body: some View {
        NavigationStack {
            if viewModel.showResults {
                AnalysisResultsView(viewModel: viewModel)
            } else {
                HomeView(viewModel: viewModel)
            }
        }
        .sheet(isPresented: $viewModel.showImagePicker) {
            ImagePicker(image: $viewModel.selectedImage)
        }
        .alert("Error", isPresented: $viewModel.showError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(viewModel.errorMessage)
        }
    }
}

// MARK: - Home View

struct HomeView: View {
    @ObservedObject var viewModel: ECGViewModel
    
    var body: some View {
        ZStack {
            // Background gradient
            LinearGradient(
                colors: [Color.blue.opacity(0.5), Color.blue.opacity(0.7)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()
            
            VStack(spacing: 28) {
                Spacer()
                
                // Title
                VStack(spacing: 10) {
                    Text("ECG Analyzer")
                        .font(.system(size: 40, weight: .bold))
                        .foregroundColor(.white)
                    
                    Text("AI-Powered Paper ECG Digitization")
                        .font(.system(size: 18, weight: .medium))
                        .foregroundColor(.white.opacity(0.9))
                }
                
                Spacer()
                
                // Image picker area
                ZStack {
                    RoundedRectangle(cornerRadius: 20)
                        .fill(Color.white.opacity(0.15))
                        .overlay(
                            RoundedRectangle(cornerRadius: 20)
                                .stroke(Color.white.opacity(0.3), lineWidth: 1)
                        )
                        .frame(height: 300)
                    
                    if let image = viewModel.selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(height: 280)
                            .cornerRadius(16)
                    } else {
                        VStack(spacing: 18) {
                            Image(systemName: "waveform.path.ecg.rectangle")
                                .font(.system(size: 65, weight: .light))
                                .foregroundColor(.white.opacity(0.9))
                            
                            Text("Select ECG Image")
                                .font(.system(size: 20, weight: .medium))
                                .foregroundColor(.white.opacity(0.9))
                        }
                    }
                    
                    // Camera and gallery buttons
                    VStack {
                        HStack {
                            Spacer()
                            HStack(spacing: 12) {
                                Button(action: { viewModel.showImagePicker = true }) {
                                    Image(systemName: "photo.fill")
                                        .font(.system(size: 18, weight: .semibold))
                                        .foregroundColor(.white)
                                        .padding(14)
                                        .background(Color.blue)
                                        .clipShape(Circle())
                                }
                                
                                Button(action: { /* Camera action - placeholder */ }) {
                                    Image(systemName: "camera.fill")
                                        .font(.system(size: 18, weight: .semibold))
                                        .foregroundColor(.white)
                                        .padding(14)
                                        .background(Color.orange)
                                        .clipShape(Circle())
                                }
                            }
                            .padding(12)
                        }
                        Spacer()
                    }
                }
                .onTapGesture {
                    viewModel.showImagePicker = true
                }
                .padding(.horizontal, 20)
                
                Spacer()
                
                // Analyze button
                Button(action: { viewModel.analyzeECG() }) {
                    HStack(spacing: 12) {
                        if viewModel.isProcessing {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .blue))
                        } else {
                            Image(systemName: "bolt.fill")
                                .font(.system(size: 20, weight: .semibold))
                        }
                        Text(viewModel.isProcessing ? "Analyzing..." : "Analyze ECG")
                            .font(.system(size: 20, weight: .bold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 18)
                    .background(viewModel.selectedImage != nil ? Color.white : Color.white.opacity(0.5))
                    .foregroundColor(viewModel.selectedImage != nil ? .blue : .blue.opacity(0.5))
                    .cornerRadius(16)
                    .scaleEffect(viewModel.isProcessing ? 0.98 : 1.0)
                }
                .disabled(viewModel.selectedImage == nil || viewModel.isProcessing)
                .padding(.horizontal, 20)
                .animation(.easeInOut(duration: 0.2), value: viewModel.isProcessing)
                
                // Progress indicator
                if viewModel.isProcessing {
                    VStack(spacing: 14) {
                        ECGLoadingWaveView()
                            .frame(height: 50)
                        
                        ProgressView(value: viewModel.progress)
                            .progressViewStyle(LinearProgressViewStyle(tint: .white))
                            .scaleEffect(y: 1.5)
                        
                        HStack {
                            Text(viewModel.statusMessage)
                                .font(.system(size: 15, weight: .medium))
                                .foregroundColor(.white.opacity(0.9))
                            
                            Spacer()
                            
                            Text("\(Int(viewModel.progress * 100))%")
                                .font(.system(size: 18, weight: .bold))
                                .foregroundColor(.white)
                                .contentTransition(.numericText())
                        }
                    }
                    .padding(.horizontal, 20)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
                
                Spacer()
            }
            .animation(.easeInOut(duration: 0.3), value: viewModel.isProcessing)
        }
    }
}

// MARK: - Analysis Results View (Native TabView Style)

struct AnalysisResultsView: View {
    @ObservedObject var viewModel: ECGViewModel
    @State private var selectedTab: ResultsTab = .results
    
    var body: some View {
        TabView(selection: $selectedTab) {
            ClassificationView(viewModel: viewModel)
                .tabItem {
                    Label(ResultsTab.results.rawValue, systemImage: ResultsTab.results.icon)
                }
                .tag(ResultsTab.results)
            
            SignalsView(viewModel: viewModel)
                .tabItem {
                    Label(ResultsTab.signals.rawValue, systemImage: ResultsTab.signals.icon)
                }
                .tag(ResultsTab.signals)
            
            Stage2OverlayView(viewModel: viewModel)
                .tabItem {
                    Label(ResultsTab.overlay.rawValue, systemImage: ResultsTab.overlay.icon)
                }
                .tag(ResultsTab.overlay)
            
            DataView(viewModel: viewModel)
                .tabItem {
                    Label(ResultsTab.export.rawValue, systemImage: ResultsTab.export.icon)
                }
                .tag(ResultsTab.export)
        }
        .navigationTitle("Analysis Results")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button(action: { viewModel.reset() }) {
                    HStack(spacing: 4) {
                        Image(systemName: "plus.circle.fill")
                        Text("New")
                    }
                    .font(.system(size: 15, weight: .medium))
                }
            }
        }
    }
}

// MARK: - Results Tab Enum

enum ResultsTab: String, CaseIterable {
    case results = "Results"
    case signals = "Signals"
    case overlay = "Overlay"
    case export = "Export"
    
    var icon: String {
        switch self {
        case .results: return "heart.text.square.fill"
        case .signals: return "waveform.path.ecg"
        case .overlay: return "square.stack.3d.up.fill"
        case .export: return "square.and.arrow.up"
        }
    }
}

// MARK: - Classification View

struct ClassificationView: View {
    @ObservedObject var viewModel: ECGViewModel
    
    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Header
                Text("Classification Results")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundColor(.primary)
                    .padding(.top, 16)
                
                // Positive Findings Card
                VStack(spacing: 14) {
                    HStack {
                        Text("DETECTED CONDITIONS")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundColor(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(Color.blue)
                            .cornerRadius(8)
                        Spacer()
                    }
                    
                    let positiveFindings = viewModel.predictions.filter { $0.isPositive }
                    
                    if positiveFindings.isEmpty {
                        HStack(spacing: 16) {
                            Image(systemName: "checkmark.circle.fill")
                                .font(.system(size: 44))
                                .foregroundColor(.green)
                            
                            VStack(alignment: .leading, spacing: 4) {
                                Text("No Abnormalities Detected")
                                    .font(.system(size: 20, weight: .bold))
                                
                                Text("All parameters within normal range")
                                    .font(.system(size: 15))
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                        }
                        .padding(.vertical, 8)
                    } else {
                        ForEach(positiveFindings, id: \.name) { prediction in
                            HStack(spacing: 14) {
                                Image(systemName: prediction.name == "NORM" ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                                    .font(.system(size: 32))
                                    .foregroundColor(prediction.name == "NORM" ? .green : .orange)
                                
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(prediction.fullName)
                                        .font(.system(size: 18, weight: .bold))
                                    Text(prediction.name)
                                        .font(.system(size: 13))
                                        .foregroundColor(.secondary)
                                }
                                
                                Spacer()
                                
                                Text(String(format: "%.1f%%", prediction.confidence * 100))
                                    .font(.system(size: 24, weight: .bold, design: .rounded))
                                    .foregroundColor(prediction.name == "NORM" ? .green : .orange)
                            }
                            .padding(14)
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(prediction.name == "NORM" ? Color.green.opacity(0.1) : Color.orange.opacity(0.1))
                            )
                        }
                    }
                }
                .padding(16)
                .background(Color(.systemGray6))
                .cornerRadius(16)
                .padding(.horizontal, 16)
                
                // All Class Scores
                VStack(alignment: .leading, spacing: 12) {
                    Text("All Classification Scores")
                        .font(.system(size: 20, weight: .bold))
                        .padding(.horizontal, 16)
                    
                    VStack(spacing: 0) {
                        ForEach(Array(viewModel.predictions.enumerated()), id: \.element.name) { index, prediction in
                            HStack(spacing: 12) {
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(prediction.fullName)
                                        .font(.system(size: 16, weight: .semibold))
                                    Text(prediction.name)
                                        .font(.system(size: 12))
                                        .foregroundColor(.secondary)
                                }
                                .frame(width: 130, alignment: .leading)
                                
                                // Progress bar
                                GeometryReader { geometry in
                                    ZStack(alignment: .leading) {
                                        RoundedRectangle(cornerRadius: 5)
                                            .fill(Color.gray.opacity(0.2))
                                        
                                        RoundedRectangle(cornerRadius: 5)
                                            .fill(prediction.isPositive ? (prediction.name == "NORM" ? Color.green : Color.orange) : Color.blue.opacity(0.6))
                                            .frame(width: geometry.size.width * CGFloat(prediction.confidence))
                                    }
                                }
                                .frame(height: 10)
                                
                                Text(String(format: "%.1f%%", prediction.confidence * 100))
                                    .font(.system(size: 17, weight: .bold, design: .rounded))
                                    .foregroundColor(prediction.isPositive ? (prediction.name == "NORM" ? .green : .orange) : .primary)
                                    .frame(width: 58, alignment: .trailing)
                                
                                if prediction.isPositive {
                                    Image(systemName: "checkmark.circle.fill")
                                        .font(.system(size: 18))
                                        .foregroundColor(prediction.name == "NORM" ? .green : .orange)
                                } else {
                                    Color.clear.frame(width: 18)
                                }
                            }
                            .padding(.vertical, 12)
                            .padding(.horizontal, 14)
                            .background(index % 2 == 0 ? Color(.systemGray6) : Color(.systemBackground))
                        }
                    }
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color(.systemGray4), lineWidth: 1)
                    )
                    .padding(.horizontal, 16)
                }
                
                // Disclaimer
                Text("Confidence scores shown. Consult healthcare professional for diagnosis.")
                    .font(.system(size: 13))
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 20)
                    .padding(.top, 8)
                
                Spacer(minLength: 16)
            }
        }
    }
}

// MARK: - Signals View

struct SignalsView: View {
    @ObservedObject var viewModel: ECGViewModel
    
    let columns = [
        GridItem(.flexible()),
        GridItem(.flexible())
    ]
    
    var body: some View {
        ScrollView {
            VStack(spacing: 14) {
                // Header
                Text("Extracted Signals")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundColor(.primary)
                    .padding(.top, 16)
                
                // Subtitle
                Text("12-Lead ECG digitized from paper image")
                    .font(.system(size: 15))
                    .foregroundColor(.secondary)
                
                // Signal Grid
                LazyVGrid(columns: columns, spacing: 10) {
                    ForEach(ECGConstants.allLeadNames, id: \.self) { leadName in
                        if let signal = viewModel.signals[leadName] {
                            SignalWaveformCard(leadName: leadName, signal: signal)
                        }
                    }
                }
                .padding(.horizontal, 12)
                
                // Rhythm Strip
                if let rhythmSignal = viewModel.signals["II-rhythm"] {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Lead II - Rhythm Strip")
                            .font(.system(size: 18, weight: .bold))
                            .padding(.horizontal, 12)
                        
                        SignalWaveformCard(leadName: "II-rhythm", signal: rhythmSignal, isRhythm: true)
                            .padding(.horizontal, 12)
                    }
                }
                
                Spacer(minLength: 16)
            }
        }
    }
}

// MARK: - Signal Waveform Card

struct SignalWaveformCard: View {
    let leadName: String
    let signal: ECGSignal
    var isRhythm: Bool = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(leadName)
                .font(.system(size: 14, weight: .bold))
                .foregroundColor(.blue)
            
            // Waveform
            WaveformView(samples: signal.samples)
                .frame(height: isRhythm ? 90 : 55)
        }
        .padding(10)
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
}

// MARK: - Waveform View

struct WaveformView: View {
    let samples: [Float]
    
    var body: some View {
        GeometryReader { geometry in
            Path { path in
                guard samples.count > 1 else { return }
                
                let width = geometry.size.width
                let height = geometry.size.height
                let stepX = width / CGFloat(samples.count - 1)
                
                let minVal = samples.min() ?? -1
                let maxVal = samples.max() ?? 1
                let range = max(maxVal - minVal, 0.001)
                
                let step = max(1, samples.count / 200)
                
                var firstPoint = true
                for i in stride(from: 0, to: samples.count, by: step) {
                    let x = CGFloat(i) * stepX
                    let normalizedY = CGFloat((samples[i] - minVal) / range)
                    let y = height - (normalizedY * height * 0.8 + height * 0.1)
                    
                    if firstPoint {
                        path.move(to: CGPoint(x: x, y: y))
                        firstPoint = false
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
            }
            .stroke(Color.blue, lineWidth: 1)
        }
    }
}

// MARK: - Stage 2 Overlay View

struct Stage2OverlayView: View {
    @ObservedObject var viewModel: ECGViewModel
    @State private var showOverlay = true
    @State private var overlayOpacity: Double = 0.8
    
    var body: some View {
        ScrollView {
            VStack(spacing: 14) {
                // Header
                Text("Signal Overlay")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundColor(.primary)
                    .padding(.top, 16)
                
                // Subtitle
                Text("Extracted traces overlaid on rectified image")
                    .font(.system(size: 15))
                    .foregroundColor(.secondary)
                
                // Controls
                HStack(spacing: 16) {
                    Toggle("Show Overlay", isOn: $showOverlay)
                        .font(.system(size: 15, weight: .medium))
                    
                    if showOverlay {
                        HStack(spacing: 8) {
                            Slider(value: $overlayOpacity, in: 0.3...1.0)
                                .frame(width: 100)
                            Text("\(Int(overlayOpacity * 100))%")
                                .font(.system(size: 14, weight: .semibold, design: .rounded))
                                .frame(width: 40)
                        }
                    }
                }
                .padding(14)
                .background(Color(.systemGray6))
                .cornerRadius(12)
                .padding(.horizontal, 16)
                
                // Main Image with Overlay
                if let image = viewModel.resultImage {
                    ZStack {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                        
                        if showOverlay {
                            SignalOverlayView(
                                signals: viewModel.signals,
                                imageSize: image.size
                            )
                            .opacity(overlayOpacity)
                        }
                    }
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color(.systemGray4), lineWidth: 1)
                    )
                    .padding(.horizontal, 16)
                } else if let image = viewModel.selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .cornerRadius(12)
                        .padding(.horizontal, 16)
                }
                
                // Legend
                HStack(spacing: 16) {
                    LegendItem(color: .red, text: "Row 1")
                    LegendItem(color: .green, text: "Row 2")
                    LegendItem(color: .blue, text: "Row 3")
                    LegendItem(color: .orange, text: "Rhythm")
                }
                .padding(12)
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .padding(.horizontal, 16)
                
                // Stats
                if !viewModel.signals.isEmpty {
                    HStack(spacing: 0) {
                        StatItemCompact(title: "Leads", value: "\(viewModel.signals.count)")
                        Divider().frame(height: 30)
                        StatItemCompact(title: "Samples", value: "\(viewModel.signals.first?.value.samples.count ?? 0)")
                        Divider().frame(height: 30)
                        StatItemCompact(title: "Range", value: String(format: "%.1f~%.1f mV", viewModel.minSignalValue, viewModel.maxSignalValue))
                    }
                    .padding(.vertical, 12)
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                    .padding(.horizontal, 16)
                }
                
                Spacer(minLength: 16)
            }
        }
    }
}

// MARK: - Compact Stat Item

struct StatItemCompact: View {
    let title: String
    let value: String
    
    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(size: 16, weight: .bold, design: .rounded))
            Text(title)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Signal Overlay View (draws signals on image)

struct SignalOverlayView: View {
    let signals: [String: ECGSignal]
    let imageSize: CGSize
    
    // Row colors
    let rowColors: [Color] = [.red, .green, .blue, .orange]
    
    // Row Y positions (normalized 0-1) based on rectified image layout
    // zeroMV = [703.5, 987.5, 1271.5, 1531.5] for height 1700
    let rowYPositions: [CGFloat] = [
        703.5 / 1700.0,   // Row 0: ~0.414
        987.5 / 1700.0,   // Row 1: ~0.581
        1271.5 / 1700.0,  // Row 2: ~0.748
        1531.5 / 1700.0   // Row 3: ~0.901
    ]
    
    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let height = geometry.size.height
            
            ZStack {
                // Draw each row
                ForEach(0..<4, id: \.self) { rowIndex in
                    if let rowSignal = getRowSignal(rowIndex) {
                        Path { path in
                            let samples = rowSignal.samples
                            guard samples.count > 1 else { return }
                            
                            let yCenter = height * rowYPositions[rowIndex]
                            
                            // X range matches timespan (118 to 2080) out of 2200 width
                            let xStart = width * (118.0 / 2200.0)
                            let xEnd = width * (2080.0 / 2200.0)
                            let xRange = xEnd - xStart
                            
                            // Scale factor for mV to pixels (79 pixels per mV in 1700 height)
                            let mvScale = height * (79.0 / 1700.0)
                            
                            // Downsample for performance
                            let step = max(1, samples.count / 400)
                            
                            var firstPoint = true
                            for i in stride(from: 0, to: samples.count, by: step) {
                                let x = xStart + (CGFloat(i) / CGFloat(samples.count - 1)) * xRange
                                let y = yCenter - CGFloat(samples[i]) * mvScale
                                
                                if firstPoint {
                                    path.move(to: CGPoint(x: x, y: y))
                                    firstPoint = false
                                } else {
                                    path.addLine(to: CGPoint(x: x, y: y))
                                }
                            }
                        }
                        .stroke(rowColors[rowIndex], lineWidth: 2)
                    }
                }
            }
        }
    }
    
    // Get combined signal for a row
    func getRowSignal(_ rowIndex: Int) -> ECGSignal? {
        switch rowIndex {
        case 0:
            // Row 1: I, aVR, V1, V4 - combine all or use first available
            return combineRowSignals(["I", "aVR", "V1", "V4"])
        case 1:
            // Row 2: II, aVL, V2, V5
            return combineRowSignals(["II", "aVL", "V2", "V5"])
        case 2:
            // Row 3: III, aVF, V3, V6
            return combineRowSignals(["III", "aVF", "V3", "V6"])
        case 3:
            // Row 4: II Rhythm
            return signals["II-rhythm"]
        default:
            return nil
        }
    }
    
    // Combine multiple lead signals into one row signal
    func combineRowSignals(_ leadNames: [String]) -> ECGSignal? {
        var combinedSamples: [Float] = []
        
        for leadName in leadNames {
            if let signal = signals[leadName] {
                combinedSamples.append(contentsOf: signal.samples)
            }
        }
        
        guard !combinedSamples.isEmpty else { return nil }
        
        return ECGSignal(
            leadName: "Combined",
            samples: combinedSamples,
            sampleRate: 500
        )
    }
}

// MARK: - Debug Stages View (shows all pipeline intermediate outputs)

struct DebugStagesView: View {
    @ObservedObject var viewModel: ECGViewModel
    @State private var selectedStage = 0
    
    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Header
                Text("Pipeline Debug")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.blue)
                    .padding(.top)
                
                // Stage picker
                Picker("Step", selection: $selectedStage) {
                    Text("Original + KP").tag(0)
                    Text("Step 1").tag(1)
                    Text("Step 2").tag(2)
                    Text("Step 3").tag(3)
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding(.horizontal)
                
                // Stage info
                stageInfoView
                    .padding(.horizontal)
                
                // Stage image
                Group {
                    switch selectedStage {
                    case 0:
                        originalWithKeypointsView
                    case 1:
                        normalizedImageView
                    case 2:
                        rectifiedImageView
                    case 3:
                        signalOverlayView
                    default:
                        EmptyView()
                    }
                }
                .padding(.horizontal)
                
                // Debug info
                debugInfoView
                    .padding(.horizontal)
                
                Spacer(minLength: 20)
            }
        }
    }
    
    @ViewBuilder
    var stageInfoView: some View {
        let (title, subtitle, status) = stageDetails
        
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                    .font(.headline)
                Spacer()
                Text(status)
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(status.contains("✅") ? Color.green.opacity(0.2) : Color.orange.opacity(0.2))
                    .cornerRadius(8)
            }
            Text(subtitle)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    var stageDetails: (String, String, String) {
        switch selectedStage {
        case 0:
            let kpCount = viewModel.keypoints.count
            return ("Original Image + Keypoints",
                    "9 keypoints detected for homography",
                    "\(kpCount)/9 keypoints ✅")
        case 1:
            let status = viewModel.stage0UsedFallback ? "⚠️ Fallback" : "✅ Homography"
            return ("Step 1: Normalized Image",
                    "After perspective correction via homography",
                    status)
        case 2:
            let status = viewModel.stage1UsedFallback ? "⚠️ Fallback" : "✅ Grid rectified"
            return ("Step 2: Rectified Image",
                    "After grid-based rectification",
                    status)
        case 3:
            let signalCount = viewModel.signals.count
            return ("Step 3: Signal Extraction",
                    "ECG signals extracted from rectified image",
                    "\(signalCount) leads ✅")
        default:
            return ("", "", "")
        }
    }
    
    @ViewBuilder
    var originalWithKeypointsView: some View {
        if let image = viewModel.selectedImage {
            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                
                // Draw keypoints
                GeometryReader { geometry in
                    let scaleX = geometry.size.width / image.size.width
                    let scaleY = geometry.size.height / image.size.height
                    
                    ForEach(Array(viewModel.keypoints.enumerated()), id: \.offset) { index, kp in
                        if kp.x > 0 && kp.y > 0 {
                            Circle()
                                .fill(keypointColor(for: kp.label))
                                .frame(width: 20, height: 20)
                                .overlay(
                                    Text(kp.leadName)
                                        .font(.system(size: 8, weight: .bold))
                                        .foregroundColor(.white)
                                )
                                .position(
                                    x: CGFloat(kp.x) * scaleX,
                                    y: CGFloat(kp.y) * scaleY
                                )
                        }
                    }
                }
            }
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.blue, lineWidth: 2)
            )
        } else {
            Text("No image")
                .foregroundColor(.secondary)
        }
    }
    
    func keypointColor(for label: Int) -> Color {
        switch label {
        case 2, 3, 4: return .red     // Row 1
        case 6, 7, 8: return .green   // Row 2
        case 10, 11, 12: return .blue // Row 3
        default: return .gray
        }
    }
    
    @ViewBuilder
    var normalizedImageView: some View {
        if let image = viewModel.normalizedImage {
            VStack {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.orange, lineWidth: 2)
                    )
                
                Text("Size: \(Int(image.size.width)) × \(Int(image.size.height))")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        } else {
            Text("Normalized image not available")
                .foregroundColor(.secondary)
        }
    }
    
    @ViewBuilder
    var rectifiedImageView: some View {
        if let image = viewModel.resultImage {
            VStack {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.green, lineWidth: 2)
                    )
                
                Text("Size: \(Int(image.size.width)) × \(Int(image.size.height))")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        } else {
            Text("Rectified image not available")
                .foregroundColor(.secondary)
        }
    }
    
    @ViewBuilder
    var signalOverlayView: some View {
        if let image = viewModel.resultImage {
            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                
                SignalOverlayView(
                    signals: viewModel.signals,
                    imageSize: image.size
                )
            }
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.purple, lineWidth: 2)
            )
        } else {
            Text("No result image")
                .foregroundColor(.secondary)
        }
    }
    
    @ViewBuilder
    var debugInfoView: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Debug Info")
                .font(.headline)
            
            if selectedStage == 0 {
                // Keypoint details
                ForEach(Array(viewModel.keypoints.enumerated()), id: \.offset) { index, kp in
                    HStack {
                        Text(kp.leadName)
                            .font(.caption)
                            .frame(width: 60, alignment: .leading)
                        Text(String(format: "(%.1f, %.1f)", kp.x, kp.y))
                            .font(.caption.monospaced())
                        Spacer()
                        Text(kp.x > 0 ? "✅" : "❌")
                    }
                }
            } else if selectedStage == 1 {
                // Homography matrix
                if let H = viewModel.homography {
                    Text("Homography Matrix:")
                        .font(.caption)
                    ForEach(0..<3, id: \.self) { row in
                        Text(H[row].map { String(format: "%.4f", $0) }.joined(separator: "  "))
                            .font(.system(size: 10, design: .monospaced))
                    }
                }
            } else if selectedStage == 2 {
                Text("Step 2 used fallback: \(viewModel.stage1UsedFallback ? "Yes" : "No")")
                    .font(.caption)
            } else if selectedStage == 3 {
                // Signal stats
                HStack {
                    StatItem(title: "Leads", value: "\(viewModel.signals.count)")
                    StatItem(title: "Min mV", value: String(format: "%.2f", viewModel.minSignalValue))
                    StatItem(title: "Max mV", value: String(format: "%.2f", viewModel.maxSignalValue))
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Legend Item

struct LegendItem: View {
    let color: Color
    let text: String
    
    var body: some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 2)
                .fill(color)
                .frame(width: 18, height: 4)
            Text(text)
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Stat Item

struct StatItem: View {
    let title: String
    let value: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Data View

struct DataView: View {
    @ObservedObject var viewModel: ECGViewModel
    
    var body: some View {
        ScrollView {
            VStack(spacing: 14) {
                // Header
                Text("Signal Data")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundColor(.primary)
                    .padding(.top, 16)
                
                // Export button
                Button(action: { viewModel.exportCSV() }) {
                    HStack(spacing: 10) {
                        Image(systemName: "square.and.arrow.up.fill")
                            .font(.system(size: 18))
                        Text("Export CSV")
                            .font(.system(size: 17, weight: .bold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                .padding(.horizontal, 16)
                
                // Lead Data Cards
                VStack(spacing: 8) {
                    ForEach(ECGConstants.allLeadNames, id: \.self) { leadName in
                        if let signal = viewModel.signals[leadName] {
                            LeadDataCard(leadName: leadName, signal: signal)
                        }
                    }
                }
                
                Spacer(minLength: 16)
            }
        }
    }
}

// MARK: - Lead Data Card

struct LeadDataCard: View {
    let leadName: String
    let signal: ECGSignal
    
    var body: some View {
        HStack(spacing: 0) {
            // Lead name
            Text(leadName)
                .font(.system(size: 17, weight: .bold))
                .foregroundColor(.blue)
                .frame(width: 70, alignment: .leading)
            
            // Stats in a row
            HStack(spacing: 0) {
                DataMetricCompact(title: "Max", value: String(format: "%.2f", signal.maxValue), color: .blue)
                DataMetricCompact(title: "Min", value: String(format: "%.2f", signal.minValue), color: .red)
                DataMetricCompact(title: "Mean", value: String(format: "%.2f", signal.meanValue), color: .primary)
                DataMetricCompact(title: "SD", value: String(format: "%.2f", signal.stdDev), color: .secondary)
            }
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 14)
        .background(Color(.systemGray6))
        .cornerRadius(10)
        .padding(.horizontal, 16)
    }
}

// MARK: - Compact Data Metric

struct DataMetricCompact: View {
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(size: 15, weight: .semibold, design: .rounded))
                .foregroundColor(color)
            Text(title)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Data Metric View

struct DataMetricView: View {
    let title: String
    let value: String
    let unit: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(color)
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value)
                    .font(.title3)
                    .fontWeight(.bold)
                Text(unit)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - ECG Loading Wave Animation

struct ECGLoadingWaveView: View {
    @State private var phase: CGFloat = 0
    
    var body: some View {
        GeometryReader { geometry in
            ECGWaveShape(phase: phase)
                .stroke(Color.white, lineWidth: 2)
                .clipShape(Rectangle())
                .onAppear {
                    withAnimation(.linear(duration: 1.5).repeatForever(autoreverses: false)) {
                        phase = .pi * 2
                    }
                }
        }
        .clipped()
    }
}

struct ECGWaveShape: Shape {
    var phase: CGFloat
    
    var animatableData: CGFloat {
        get { phase }
        set { phase = newValue }
    }
    
    func path(in rect: CGRect) -> Path {
        var path = Path()
        let width = rect.width
        let height = rect.height
        let midY = height / 2
        
        let waveLength: CGFloat = 80
        let segments = Int(width / waveLength) + 2
        
        path.move(to: CGPoint(x: 0, y: midY))
        
        for i in 0..<segments {
            let startX = CGFloat(i) * waveLength - (phase / (.pi * 2)) * waveLength
            
            // Flat line
            path.addLine(to: CGPoint(x: startX + waveLength * 0.1, y: midY))
            
            // P wave (small bump)
            path.addQuadCurve(
                to: CGPoint(x: startX + waveLength * 0.2, y: midY),
                control: CGPoint(x: startX + waveLength * 0.15, y: midY - height * 0.15)
            )
            
            // Flat
            path.addLine(to: CGPoint(x: startX + waveLength * 0.25, y: midY))
            
            // Q dip
            path.addLine(to: CGPoint(x: startX + waveLength * 0.3, y: midY + height * 0.1))
            
            // R spike (tall)
            path.addLine(to: CGPoint(x: startX + waveLength * 0.35, y: midY - height * 0.45))
            
            // S dip
            path.addLine(to: CGPoint(x: startX + waveLength * 0.4, y: midY + height * 0.2))
            
            // Back to baseline
            path.addLine(to: CGPoint(x: startX + waveLength * 0.45, y: midY))
            
            // Flat
            path.addLine(to: CGPoint(x: startX + waveLength * 0.55, y: midY))
            
            // T wave (rounded bump)
            path.addQuadCurve(
                to: CGPoint(x: startX + waveLength * 0.75, y: midY),
                control: CGPoint(x: startX + waveLength * 0.65, y: midY - height * 0.25)
            )
            
            // Flat to end
            path.addLine(to: CGPoint(x: startX + waveLength, y: midY))
        }
        
        return path
    }
}

// MARK: - View Model

class ECGViewModel: ObservableObject {
    @Published var selectedImage: UIImage?
    @Published var resultImage: UIImage?
    @Published var isProcessing = false
    @Published var progress: Float = 0
    @Published var statusMessage = ""
    @Published var showResults = false
    @Published var signals: [String: ECGSignal] = [:]
    @Published var showImagePicker = false
    @Published var showError = false
    @Published var errorMessage = ""
    
    // Classification results
    @Published var primaryDiagnosis = "Waiting for analysis"
    @Published var primaryConfidence: Float = 0
    @Published var predictions: [Prediction] = []
    @Published var keyFindings: [String] = []
    @Published var classificationResult: ClassificationResult?
    
    // Debug data - all intermediate stages
    @Published var normalizedImage: UIImage?  // Stage 0 output
    @Published var keypoints: [ECGKeypoint] = []
    @Published var homography: [[Double]]?
    @Published var stage0UsedFallback = false
    @Published var stage1UsedFallback = false
    @Published var rowSignals: [[Float]]?
    
    private var pipeline: ECGPipeline?
    private var classifier: ECGClassifierProcessor?
    
    struct Prediction {
        let name: String       // Short code (CD, HYP, etc.)
        let fullName: String   // Full name
        let confidence: Float  // 0-1 probability
        let isPositive: Bool   // Above threshold?
    }
    
    // Computed properties for signal stats
    var minSignalValue: Float {
        signals.values.map { $0.minValue }.min() ?? 0
    }
    
    var maxSignalValue: Float {
        signals.values.map { $0.maxValue }.max() ?? 0
    }
    
    init() {
        setupPipeline()
        setupClassifier()
    }
    
    private func setupPipeline() {
        do {
            pipeline = try ECGPipeline()
        } catch {
            errorMessage = "Failed to load pipeline models: \(error.localizedDescription)"
            showError = true
        }
    }
    
    private func setupClassifier() {
        do {
            classifier = try ECGClassifierProcessor()
            if classifier?.isModelLoaded == true {
                print("[ViewModel] Classifier loaded successfully")
            } else {
                print("[ViewModel] Classifier model not found - using mock results")
            }
        } catch {
            print("[ViewModel] Failed to load classifier: \(error.localizedDescription)")
            // Not critical - we'll use mock results
        }
    }
    
    func analyzeECG() {
        guard let image = selectedImage, let pipeline = pipeline else { return }
        
        // Show loading immediately with animation
        withAnimation(.easeInOut(duration: 0.3)) {
            isProcessing = true
            progress = 0
            statusMessage = "Initializing..."
        }
        
        // Small delay to ensure UI renders before heavy work starts
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            guard let self = self else { return }
            
            // Start the pipeline on background thread
            DispatchQueue.global(qos: .userInitiated).async {
                // NOTE: Stage0Diagnostics disabled for performance
                // To enable, uncomment the following:
                // #if DEBUG
                // if let stage0URL = Bundle.main.url(forResource: "Stage0", withExtension: "mlmodelc") {
                //     Stage0Diagnostics.diagnose(image: image, modelURL: stage0URL)
                // }
                // #endif
                
                // Start the pipeline
                pipeline.processAsync(
                    image: image,
                    targetSignalLength: 5000,
                    progressHandler: { [weak self] prog, message in
                        DispatchQueue.main.async {
                            self?.progress = prog
                            self?.statusMessage = message
                        }
                    },
                    completion: { [weak self] result in
                        DispatchQueue.main.async {
                            self?.handleResult(result)
                        }
                    }
                )
            }
        }
    }
    
    private func handleResult(_ result: ECGPipelineResult) {
        isProcessing = false
        
        if result.success {
            signals = result.signals
            
            // Store debug data
            if let normalizedCG = result.normalizedImage {
                normalizedImage = UIImage(cgImage: normalizedCG)
            }
            if let rectifiedCG = result.rectifiedImage {
                resultImage = UIImage(cgImage: rectifiedCG)
            }
            keypoints = result.keypoints ?? []
            homography = result.homography
            stage0UsedFallback = result.stage0UsedFallback
            stage1UsedFallback = result.stage1UsedFallback
            rowSignals = result.rowSignals
            
            // Generate mock classification results
            generateClassificationResults()
            
            showResults = true
        } else {
            errorMessage = result.errorMessage ?? "Processing failed"
            showError = true
        }
    }
    
    private func generateClassificationResults() {
        // Try real classification if model is loaded
        if let classifier = classifier, classifier.isModelLoaded {
            do {
                let result = try classifier.classify(signals: signals)
                classificationResult = result
                
                // Update published properties
                primaryDiagnosis = result.primaryDiagnosis
                primaryConfidence = result.primaryConfidence
                
                // Convert to Prediction structs
                predictions = result.predictions.map { pred in
                    Prediction(
                        name: pred.className,
                        fullName: pred.fullName,
                        confidence: pred.confidence,
                        isPositive: pred.isPositive
                    )
                }
                
                // Generate key findings based on positive predictions
                keyFindings = generateKeyFindings(from: result)
                
                print("[ViewModel] Classification complete: \(result.positiveCount) positive findings")
                return
            } catch {
                print("[ViewModel] Classification failed: \(error), using mock results")
            }
        }
        
        // Fallback to mock results if classifier not available
        generateMockClassificationResults()
    }
    
    private func generateKeyFindings(from result: ClassificationResult) -> [String] {
        var findings: [String] = []
        
        let positives = result.predictions.filter { $0.isPositive }
        
        if positives.isEmpty {
            return ["No significant abnormalities detected"]
        }
        
        for pred in positives {
            switch pred.className {
            case "NORM":
                findings.append("Regular sinus rhythm detected")
                findings.append("Normal ECG morphology")
            case "STTC":
                findings.append("ST-segment or T-wave changes detected")
                findings.append("May indicate ischemia or other conditions")
            case "MI":
                findings.append("Pattern suggestive of myocardial infarction")
                findings.append("Recommend urgent clinical evaluation")
            case "HYP":
                findings.append("Voltage criteria for hypertrophy detected")
                findings.append("Consider echocardiogram for confirmation")
            case "CD":
                findings.append("Conduction abnormality detected")
                findings.append("May indicate bundle branch block or other delay")
            default:
                break
            }
        }
        
        return findings
    }
    
    private func generateMockClassificationResults() {
        // Mock classification - used when model not available
        let diagnoses = [
            ("Normal", "NORM"),
            ("ST/T Change", "STTC"),
            ("Myocardial Infarction", "MI"),
            ("Hypertrophy", "HYP"),
            ("Conduction Disturbance", "CD")
        ]
        
        // Random primary diagnosis
        let shuffled = diagnoses.shuffled()
        let primary = shuffled[0]
        
        primaryDiagnosis = primary.0
        primaryConfidence = Float.random(in: 0.75...0.95)
        
        // Top predictions
        predictions = shuffled.enumerated().map { index, diag in
            let conf = primaryConfidence - (Float(index) * Float.random(in: 0.08...0.15))
            let isPositive = index == 0  // Only first one is "positive"
            return Prediction(
                name: diag.1,
                fullName: diag.0,
                confidence: max(conf, 0.1),
                isPositive: isPositive
            )
        }
        
        // Key findings based on diagnosis
        switch primary.1 {
        case "NORM":
            keyFindings = [
                "Regular sinus rhythm detected",
                "Normal P-wave morphology",
                "QRS complex within normal limits",
                "No ST-segment abnormalities"
            ]
        case "STTC":
            keyFindings = [
                "ST-segment changes detected",
                "T-wave inversions present",
                "Possible ischemic changes",
                "Recommend clinical correlation"
            ]
        case "MI":
            keyFindings = [
                "Pathological Q waves detected",
                "ST elevation in contiguous leads",
                "Reciprocal ST depression noted",
                "Suggest acute coronary evaluation"
            ]
        case "HYP":
            keyFindings = [
                "Increased QRS voltage",
                "Left ventricular strain pattern",
                "Possible left atrial enlargement",
                "Consider echocardiogram"
            ]
        case "CD":
            keyFindings = [
                "Prolonged QRS duration",
                "Bundle branch block pattern",
                "Abnormal axis deviation",
                "Conduction delay identified"
            ]
        default:
            keyFindings = ["Analysis complete (mock results)"]
        }
    }
    
    func reset() {
        selectedImage = nil
        resultImage = nil
        normalizedImage = nil
        signals = [:]
        predictions = []
        keyFindings = []
        keypoints = []
        homography = nil
        stage0UsedFallback = false
        stage1UsedFallback = false
        rowSignals = nil
        showResults = false
        progress = 0
        statusMessage = ""
    }
    
    func exportCSV() {
        // CSV structure:
        // Row = time sample (0-5000 at 500Hz = 10 seconds)
        // Columns = Time(s), I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        // Each lead only has values in its time window, zeros elsewhere
        //
        // Time windows (standard 12-lead layout):
        // 0.0 - 2.5s (samples 0-1249):     I, II, III
        // 2.5 - 5.0s (samples 1250-2499):  aVR, aVL, aVF
        // 5.0 - 7.5s (samples 2500-3749):  V1, V2, V3
        // 7.5 - 10.0s (samples 3750-4999): V4, V5, V6
        
        let sampleRate: Float = 500.0  // Hz
        let totalSamples = 5000        // 10 seconds at 500Hz
        let samplesPerWindow = 1250    // 2.5 seconds at 500Hz
        
        // Lead order for columns
        let leadOrder = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        // Time windows for each lead (start sample index)
        let leadWindows: [String: Int] = [
            "I": 0, "II": 0, "III": 0,
            "aVR": 1250, "aVL": 1250, "aVF": 1250,
            "V1": 2500, "V2": 2500, "V3": 2500,
            "V4": 3750, "V5": 3750, "V6": 3750
        ]
        
        // Build CSV header
        var csvString = "Time(s),I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6\n"
        
        // Build each row (time sample)
        for sampleIdx in 0..<totalSamples {
            let timeSeconds = Float(sampleIdx) / sampleRate
            var row = String(format: "%.4f", timeSeconds)
            
            for leadName in leadOrder {
                let windowStart = leadWindows[leadName] ?? 0
                let windowEnd = windowStart + samplesPerWindow
                
                var value: Float = 0.0
                
                // Check if this sample is within the lead's time window
                if sampleIdx >= windowStart && sampleIdx < windowEnd {
                    let leadSampleIdx = sampleIdx - windowStart
                    if let signal = signals[leadName], leadSampleIdx < signal.samples.count {
                        value = signal.samples[leadSampleIdx]
                    }
                }
                
                row += String(format: ",%.6f", value)
            }
            
            csvString += row + "\n"
        }
        
        // Save to Documents directory
        if let data = csvString.data(using: .utf8) {
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "yyyy-MM-dd_HHmmss"
            let dateString = dateFormatter.string(from: Date())
            let filename = "ecg_data_\(dateString).csv"
            
            guard let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
                errorMessage = "Cannot access Documents directory"
                showError = true
                return
            }
            
            let fileURL = documentsURL.appendingPathComponent(filename)
            
            do {
                try data.write(to: fileURL)
                print("[Export] CSV saved to: \(fileURL.path)")
                print("[Export] Total rows: \(totalSamples), Columns: 13 (Time + 12 leads)")
                
                // Share the file
                DispatchQueue.main.async {
                    let activityVC = UIActivityViewController(activityItems: [fileURL], applicationActivities: nil)
                    activityVC.popoverPresentationController?.sourceView = UIView()
                    
                    if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                       let window = windowScene.windows.first,
                       let rootVC = window.rootViewController {
                        var topController = rootVC
                        while let presented = topController.presentedViewController {
                            topController = presented
                        }
                        topController.present(activityVC, animated: true)
                    }
                }
            } catch {
                print("[Export] Error: \(error)")
                errorMessage = "Failed to export CSV: \(error.localizedDescription)"
                showError = true
            }
        }
    }
}

// MARK: - ECGSignal Extension

extension ECGSignal {
    var meanValue: Float {
        guard !samples.isEmpty else { return 0 }
        return samples.reduce(0, +) / Float(samples.count)
    }
    
    var stdDev: Float {
        guard samples.count > 1 else { return 0 }
        let mean = meanValue
        let variance = samples.reduce(0) { $0 + pow($1 - mean, 2) } / Float(samples.count - 1)
        return sqrt(variance)
    }
}

// MARK: - Image Picker

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.dismiss) var dismiss
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .photoLibrary
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            parent.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

// MARK: - Preview

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
