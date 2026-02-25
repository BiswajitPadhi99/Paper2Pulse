# MMECG - iOS ECG Digitizer & Classifier

An iOS application that digitizes 12-lead ECG images from paper printouts and classifies them for cardiac abnormalities using on-device machine learning.

![Platform](https://img.shields.io/badge/Platform-iOS%2015%2B-blue)
![Swift](https://img.shields.io/badge/Swift-5.9-orange)
![CoreML](https://img.shields.io/badge/CoreML-ML%20Program-green)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
<!-- - [Project Structure](#project-structure)
- [Component Details](#component-details)
- [Running the App](#running-the-app)
- [Benchmarking](#benchmarking)
- [Disabling Benchmarking](#disabling-benchmarking-for-production)
- [Model Information](#model-information)
- [Troubleshooting](#troubleshooting) -->

---

## Overview

MMECG captures or loads an image of a standard 12-lead ECG printout, extracts the waveform signals digitally, and runs a multi-label classifier to detect cardiac abnormalities—all on-device without requiring internet connectivity.

### Pipeline Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   📷 Image   │───▶│   Stage 0    │───▶│   Stage 1    │───▶│   Stage 2    │
│   Capture    │    │  Keypoints   │    │    Grid      │    │   Signals    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                    ┌──────────────┐    ┌──────────────┐            │
                    │   📊 Results │◀───│  Classifier  │◀───────────┘
                    │   Display    │    │   Model      │
                    └──────────────┘    └──────────────┘
```

---

## Features

- 📷 **Image Capture** - Take photo or select from gallery
- 🔍 **Automatic Detection** - Detects ECG grid corners and structure
- 📈 **Signal Digitization** - Extracts 12-lead waveforms at 500Hz
- 🏥 **AI Classification** - Multi-label cardiac abnormality detection
- 📱 **Fully On-Device** - No internet required, ensures data privacy
- 📊 **Benchmarking Tools** - Performance measurement across devices
- 💾 **Export Options** - Save signals as CSV, share results

---

## Requirements

### Development
- macOS 14.0 (Sonoma) or later
- Xcode 15.0 or later
- Git LFS (for large model files)

### Runtime
- iOS 15.0 or later
- iPhone or iPad with A12 Bionic chip or later (for Neural Engine)

---

## Installation

### 1. Clone the Repository

```bash
# Install Git LFS first (required for model files)
brew install git-lfs
git lfs install

# Clone the repository
git clone https://github.com/BiswajitPadhi99/mmecg-ios-app.git
cd mmecg-ios-app
```

### 2. Open in Xcode

```bash
open MMECG/MMECG.xcodeproj
```

### 3. Configure Signing

1. Select the **MMECG** project in the navigator
2. Select the **MMECG** target
3. Go to **Signing & Capabilities** tab
4. Select your **Team** from the dropdown
5. Ensure **Automatically manage signing** is checked

### 4. Build and Run

1. Connect your iOS device via USB
2. Select your device from the device dropdown (top of Xcode)
3. Press **⌘R** or click the **Run** button
4. Trust the developer certificate on your device if prompted:
   - Go to **Settings → General → VPN & Device Management**
   - Tap your developer account and tap **Trust**
<!-- 
---

## Project Structure

```
MMECG/
├── MMECG.xcodeproj          # Xcode project file
├── MMECG/
│   ├── MMECGApp.swift       # App entry point
│   ├── ContentView.swift    # Main UI and ViewModel
│   │
│   ├── Pipeline/            # Core ML processing pipeline
│   │   ├── ECGPipeline.swift           # Main orchestrator
│   │   ├── Stage0Processor.swift       # Keypoint detection
│   │   ├── Stage1Processor.swift       # Grid detection
│   │   ├── Stage2Processor.swift       # Signal extraction
│   │   ├── ECGClassifierProcessor.swift # Classification
│   │   ├── HomographyComputer.swift    # Perspective transform
│   │   └── ImageUtils.swift            # Image helpers
│   │
│   ├── Benchmark/           # Performance testing
│   │   ├── ECGBenchmarker.swift        # Benchmark logic
│   │   └── BenchmarkView.swift         # Benchmark UI
│   │
│   ├── Models/              # CoreML models (compiled)
│   │   ├── Stage0.mlmodelc/            # Keypoint model
│   │   ├── Stage1.mlmodelc/            # Grid model
│   │   ├── Stage2.mlmodelc/            # Signal model
│   │   └── ECGClassifier.mlmodelc/     # Classifier model
│   │
│   ├── Constants/           # Configuration
│   │   ├── Constants.swift             # Grid dimensions
│   │   └── ClassifierConstants.swift   # Normalization stats
│   │
│   ├── TestImages/          # Benchmark test images
│   │   └── *.png
│   │
│   └── Assets.xcassets      # App icons and images
│
└── README.md                # This file
```

---

## Component Details

### 📱 App Entry Point

#### `MMECGApp.swift`
The main app structure using SwiftUI's `App` protocol. Configures the root view with tab navigation between the analyzer and benchmark views.

---

### 🎨 Views

#### `ContentView.swift`
The main user interface containing:

| Component | Description |
|-----------|-------------|
| `ContentView` | Root view with navigation |
| `HomeView` | Image selection and analyze button |
| `AnalysisResultsView` | Results display with tabs |
| `SignalView` | ECG waveform visualization |
| `ClassificationView` | Diagnosis results display |
| `ECGViewModel` | State management and pipeline coordination |

**Key Features:**
- Image picker integration
- Animated loading indicator with ECG wave
- Tab-based results (Signals, Classification, Debug)
- Export functionality

---

### ⚙️ Pipeline Components

#### `ECGPipeline.swift`
Main orchestrator that coordinates the entire processing flow.

```swift
// Usage
let pipeline = try ECGPipeline()
pipeline.processAsync(image: uiImage, targetSignalLength: 5000) { progress, message in
    // Update UI with progress
} completion: { result in
    // Handle result
}
```

#### `Stage0Processor.swift`
**Purpose:** Detects 9 keypoints on the ECG grid (3×3 grid of corners)

| Input | Output |
|-------|--------|
| 512×512 RGB image | 9 keypoints with (x, y) coordinates |

**Process:**
1. Resizes input image to 512×512
2. Normalizes pixel values
3. Runs CoreML inference
4. Extracts keypoint coordinates from heatmaps

#### `Stage1Processor.swift`
**Purpose:** Detects the ECG grid structure for perspective correction

| Input | Output |
|-------|--------|
| 800×600 grayscale image | Grid mask |

**Process:**
1. Converts to grayscale
2. Resizes to model input size
3. Detects grid lines and structure
4. Outputs binary mask of grid

#### `Stage2Processor.swift`
**Purpose:** Extracts signal values from individual ECG lead rows

| Input | Output |
|-------|--------|
| Row image (single lead) | Array of voltage values |

**Process:**
1. Takes perspective-corrected row image
2. Extracts waveform trace
3. Converts pixel positions to voltage values
4. Outputs signal array at 500Hz

#### `ECGClassifierProcessor.swift`
**Purpose:** Multi-label classification of cardiac abnormalities

| Input | Output |
|-------|--------|
| 5000×12 signal array | 5 class probabilities |

**Classes Detected:**

| Code | Full Name | Threshold |
|------|-----------|-----------|
| CD | Conduction Disturbance | 0.51 |
| HYP | Hypertrophy | 0.50 |
| MI | Myocardial Infarction | 0.51 |
| NORM | Normal | 0.54 |
| STTC | ST/T Change | 0.54 |

#### `HomographyComputer.swift`
**Purpose:** Computes perspective transformation matrix

**Key Features:**
- Uses RANSAC for robust estimation
- LAPACK-accelerated SVD computation
- Handles column-major matrix storage for Accelerate framework

#### `ImageUtils.swift`
Helper functions for image processing:
- Color space conversion
- Resizing and cropping
- Pixel buffer manipulation
- CGImage/UIImage conversion

---

### 📊 Benchmark Components

#### `ECGBenchmarker.swift`
Performance measurement system that tracks:

| Metric | Description |
|--------|-------------|
| Latency | Per-stage and end-to-end timing (ms) |
| Memory | Peak and average RAM usage (MB) |
| CPU | Processor utilization (%) |
| Storage | Model and app sizes (KB) |

**Output Formats:**
- JSON (complete report)
- CSV (detailed per-image)
- CSV (summary statistics)

#### `BenchmarkView.swift`
SwiftUI interface for running benchmarks:
- Device information display
- Model size display
- Image loading from bundle/folder
- Progress tracking with ETA
- Results summary
- Export functionality

---

### 📁 Constants

#### `Constants.swift`
Reference coordinates and dimensions:
- Grid reference points
- Image dimensions
- Lead positions
- Time windows

#### `ClassifierConstants.swift`
Classification model parameters:
```swift
struct ClassifierConstants {
    static let labelNames = ["CD", "HYP", "MI", "NORM", "STTC"]
    static let thresholds: [Float] = [0.51, 0.50, 0.51, 0.54, 0.54]
    static let signalMean: [Float] = [...]  // Per-lead normalization
    static let signalStd: [Float] = [...]   // Per-lead normalization
}
```

---

## Running the App

### On Simulator (Limited)
⚠️ **Note:** CoreML models run slowly on simulator. Use a real device for best results.

1. Select a simulator from the device dropdown
2. Press **⌘R** to build and run
3. Use "Photo Library" to load test images

### On Physical Device (Recommended)

1. **Connect device** via USB cable
2. **Select device** from dropdown in Xcode toolbar
3. **Run** with **⌘R**
4. **Trust certificate** on device if first time:
   - Settings → General → VPN & Device Management → Trust

### Using the App

1. **Select Image**
   - Tap the image area or 📷 button
   - Choose from Photo Library or take a photo

2. **Analyze**
   - Tap "Analyze ECG" button
   - Wait for processing (8-15 seconds)

3. **View Results**
   - **Signals Tab:** View digitized 12-lead waveforms
   - **Classification Tab:** View diagnosis predictions
   - **Debug Tab:** View intermediate processing stages

4. **Export**
   - Tap export button to save CSV
   - Share via AirDrop, email, or files

---

## Benchmarking

### Running Benchmarks

1. Open the app and tap the **Benchmark** tab
2. Tap **"From Bundle"** to load test images
3. Verify image count (e.g., "Loaded 100 images")
4. Tap **"Start Benchmark"**
5. Wait for completion (shows progress and ETA)
6. Review results summary
7. Tap **"Export Report"** to save

### Benchmark Outputs

| File | Description |
|------|-------------|
| `benchmark_report.json` | Complete data in JSON format |
| `benchmark_detailed.csv` | Per-image metrics |
| `benchmark_summary.csv` | Statistical summary |
| `device_info.json` | Device specifications |
| `model_sizes.json` | Model file sizes |

### Multi-Device Testing

To compare across devices:

1. Run benchmark on each device
2. Export reports from each
3. Transfer to Mac
4. Use aggregation script:

```bash
python aggregate_benchmarks.py --input ./reports --output ./analysis
```

---

## Disabling Benchmarking for Production

To remove the benchmark tab for App Store release:

### Option 1: Conditional Compilation (Recommended)

In `MMECGApp.swift`, use compiler flags:

```swift
import SwiftUI

@main
struct MMECGApp: App {
    var body: some Scene {
        WindowGroup {
            #if DEBUG
            // Development: Show both tabs
            TabView {
                ContentView()
                    .tabItem {
                        Label("Analyze", systemImage: "waveform.path.ecg")
                    }
                
                BenchmarkView()
                    .tabItem {
                        Label("Benchmark", systemImage: "chart.bar")
                    }
            }
            #else
            // Production: Only show main app
            ContentView()
            #endif
        }
    }
}
```

This shows benchmarking in **Debug** builds only, not in **Release** builds.

### Option 2: Remove Completely

1. In `MMECGApp.swift`, replace TabView with just ContentView:

```swift
import SwiftUI

@main
struct MMECGApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
```

2. Optionally delete benchmark files:
   - `Benchmark/ECGBenchmarker.swift`
   - `Benchmark/BenchmarkView.swift`
   - `TestImages/` folder

3. Remove from Xcode:
   - Right-click `Benchmark` folder → Delete → Move to Trash
   - Right-click `TestImages` folder → Delete → Move to Trash

### Option 3: Feature Flag

Add a runtime toggle in a new `AppConfig.swift`:

```swift
struct AppConfig {
    static let showBenchmarkTab = false  // Set to false for production
}
```

In `MMECGApp.swift`:

```swift
var body: some Scene {
    WindowGroup {
        if AppConfig.showBenchmarkTab {
            TabView {
                ContentView()
                    .tabItem { Label("Analyze", systemImage: "waveform.path.ecg") }
                BenchmarkView()
                    .tabItem { Label("Benchmark", systemImage: "chart.bar") }
            }
        } else {
            ContentView()
        }
    }
}
```

---

## Model Information

### Model Specifications

| Model | Input Shape | Output Shape | Size | Purpose |
|-------|-------------|--------------|------|---------|
| Stage0 | 1×3×512×512 | 9×3 | ~5 MB | Keypoint detection |
| Stage1 | 1×1×600×800 | 1×1×600×800 | ~8 MB | Grid segmentation |
| Stage2 | Variable | Variable | ~3 MB | Signal extraction |
| Classifier | 1×5000×12 | 1×5 | ~15 MB | Multi-label classification |

### Signal Specifications

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 500 Hz |
| Duration | 10 seconds |
| Total Samples | 5000 |
| Leads | 12 (I, II, III, aVR, aVL, aVF, V1-V6) |
| Units | Millivolts (mV) |

### Lead Time Windows

```
Time:    0.0s ─────── 2.5s ─────── 5.0s ─────── 7.5s ─────── 10.0s
         │            │            │            │            │
Row 1:   I ───────────┤            │            │            │
Row 2:   II ──────────────────────────────────────────────────┤
Row 3:   III ─────────┤            │            │            │
Row 4:                aVR ─────────┤            │            │
Row 5:                aVL ─────────┤            │            │
Row 6:                aVF ─────────┤            │            │
Row 7:                             V1 ──────────┤            │
Row 8:                             V2 ──────────┤            │
Row 9:                             V3 ──────────┤            │
Row 10:                                         V4 ──────────┤
Row 11:                                         V5 ──────────┤
Row 12:                                         V6 ──────────┤
```

---

## Troubleshooting

### Common Issues

#### "No such module 'CoreML'"
- Ensure deployment target is iOS 15.0+
- Clean build folder: **⌘+Shift+K**

#### Models not loading
- Verify `.mlmodelc` folders are in **Copy Bundle Resources**
- Check build phases in Xcode

#### App crashes on launch
- Check device iOS version (requires 15.0+)
- Verify all model files were cloned (use `git lfs pull`)

#### Slow performance on Simulator
- Use a physical device for testing
- Simulator doesn't use Neural Engine

#### Git LFS files not downloading
```bash
git lfs install
git lfs pull
```

#### Signing errors
- Select your development team in Signing & Capabilities
- Ensure you have a valid Apple Developer account

#### "No test images found in bundle"
- Ensure TestImages folder is added as folder reference (blue folder icon)
- Check that images are in Copy Bundle Resources build phase

### Debug Logging

View console output in Xcode:
- **⌘+Shift+C** to open console
- Filter by `[Pipeline]`, `[Stage0]`, `[Classifier]`, `[Benchmark]`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or issues, please open a GitHub issue. -->
