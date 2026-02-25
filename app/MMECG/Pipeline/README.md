# iOS ECG Digitization Pipeline

Pure Swift implementation of the ECG image digitization pipeline using Apple native frameworks.

## Overview

This pipeline converts a photograph of an ECG paper into digital signals (12 leads + II rhythm strip).

```
Raw ECG Image
      ↓
┌─────────────────┐
│    Stage 0      │ → Orientation detection + Keypoint extraction
│   (CoreML)      │ → Homography computation → Perspective warp
└─────────────────┘
      ↓
Normalized Image (1152×1440)
      ↓
┌─────────────────┐
│    Stage 1      │ → Grid line detection
│   (CoreML)      │ → Grid point extraction → Image rectification
└─────────────────┘
      ↓
Rectified Image (1700×2200)
      ↓
┌─────────────────┐
│    Stage 2      │ → Signal trace detection
│   (CoreML)      │ → Y-position extraction → mV conversion
└─────────────────┘
      ↓
12 Lead ECG Signals (mV)
```

## Requirements

- iOS 14.0+
- Xcode 13.0+
- iPhone 12+ (for optimal Neural Engine performance)

## Setup

### 1. Add CoreML Models

Copy the compiled CoreML models to your Xcode project:

```
YourApp/
├── Models/
│   ├── Stage0.mlmodelc
│   ├── Stage1.mlmodelc
│   └── Stage2.mlmodelc
```

To compile `.mlpackage` to `.mlmodelc`:
```bash
xcrun coremlcompiler compile Stage0.mlpackage ./
xcrun coremlcompiler compile Stage1.mlpackage ./
xcrun coremlcompiler compile Stage2.mlpackage ./
```

### 2. Add Source Files

Add all Swift files to your project:
- `Constants.swift`
- `ConnectedComponents.swift`
- `HomographyComputer.swift`
- `ImageUtils.swift`
- `GridSampler.swift`
- `Stage0Processor.swift`
- `Stage1Processor.swift`
- `Stage2Processor.swift`
- `ECGPipeline.swift`
- `ECGViewController.swift` (optional, for UI)

### 3. Link Frameworks

Ensure these frameworks are linked (all are included in iOS SDK):
- CoreML
- Accelerate
- CoreImage
- UIKit

## Usage

### Basic Usage

```swift
import UIKit

// Initialize pipeline (loads models from bundle)
do {
    let pipeline = try ECGPipeline()
    
    // Process an image
    let ecgImage = UIImage(named: "ecg_photo")!
    let result = pipeline.process(image: ecgImage)
    
    if result.success {
        // Access 12-lead signals
        for leadName in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"] {
            if let signal = result.signals[leadName] {
                print("\(leadName): \(signal.samples.count) samples")
                print("  Range: [\(signal.minValue), \(signal.maxValue)] mV")
            }
        }
        
        // Access II rhythm strip
        if let rhythm = result.signals["II-rhythm"] {
            print("II-rhythm: \(rhythm.samples.count) samples over \(rhythm.duration)s")
        }
    } else {
        print("Error: \(result.errorMessage ?? "Unknown")")
    }
} catch {
    print("Failed to initialize pipeline: \(error)")
}
```

### Async Processing with Progress

```swift
pipeline.processAsync(
    image: ecgImage,
    targetSignalLength: 5000,
    progressHandler: { progress, message in
        print("[\(Int(progress * 100))%] \(message)")
    },
    completion: { result in
        // Handle result on main thread
    }
)
```

### Custom Model Locations

```swift
let stage0URL = URL(fileURLWithPath: "/path/to/Stage0.mlmodelc")
let stage1URL = URL(fileURLWithPath: "/path/to/Stage1.mlmodelc")
let stage2URL = URL(fileURLWithPath: "/path/to/Stage2.mlmodelc")

let pipeline = try ECGPipeline(
    stage0URL: stage0URL,
    stage1URL: stage1URL,
    stage2URL: stage2URL
)
```

### Debug Mode

```swift
let (result, debugInfo) = pipeline.processWithDebug(image: ecgImage)

print("Stage 0 time: \(debugInfo["stage0_time"] ?? 0)s")
print("Stage 1 time: \(debugInfo["stage1_time"] ?? 0)s")
print("Stage 2 time: \(debugInfo["stage2_time"] ?? 0)s")
print("Total time: \(debugInfo["total_time"] ?? 0)s")
```

## Output Format

### ECGSignal Structure

```swift
struct ECGSignal {
    let leadName: String      // e.g., "I", "V1", "II-rhythm"
    let samples: [Float]      // Signal values in millivolts
    let sampleRate: Float     // Samples per second
    
    var duration: Float       // Duration in seconds
    var minValue: Float       // Minimum mV value
    var maxValue: Float       // Maximum mV value
}
```

### Signal Ranges

Expected signal values for healthy ECG:
- Typical range: -2.0 to +2.0 mV
- Maximum range: -5.0 to +5.0 mV (clipped)

### Lead Names

Standard 12-lead ECG:
- Limb leads: I, II, III
- Augmented leads: aVR, aVL, aVF
- Precordial leads: V1, V2, V3, V4, V5, V6
- Rhythm strip: II-rhythm (10-second continuous)

## Constants

Key parameters (defined in `Constants.swift`):

```swift
// Signal extraction parameters
zeroMV = [703.5, 987.5, 1271.5, 1531.5]  // Y-position of 0mV per row
mvToPixel = 79.0                          // Pixels per millivolt
timespan = (118, 2080)                    // X-range for extraction

// Image dimensions
normalizedSize = (1440, 1152)             // After Stage 0
rectifiedSize = (2200, 1700)              // After Stage 1
stage2InputSize = (2176, 1696)            // Stage 2 input (cropped)
```

## Architecture

### Pure Apple Native

- **CoreML** - Neural network inference
- **Accelerate** - SVD for homography (LAPACK)
- **CoreImage** - (Optional) Image transforms
- **UIKit/CoreGraphics** - Image manipulation

### No External Dependencies

Unlike the Python version which uses:
- OpenCV → Replaced with native code
- cc3d → Replaced with `ConnectedComponents.swift`
- PyTorch → Replaced with CoreML
- scipy → Replaced with native interpolation

## Performance

On iPhone 12 Pro:
- Stage 0: ~200ms
- Stage 1: ~300ms
- Stage 2: ~400ms
- **Total: ~1 second**

On iPhone 14 Pro (with Neural Engine):
- **Total: ~500ms**

## Troubleshooting

### Models Not Found

Ensure models are in the app bundle:
```swift
// Check if models exist
let stage0Path = Bundle.main.path(forResource: "Stage0", ofType: "mlmodelc")
print("Stage0 path: \(stage0Path ?? "NOT FOUND")")
```

### Low Quality Results

1. Ensure input image is well-lit and in focus
2. ECG should fill most of the image
3. Avoid shadows and glare
4. Standard 12-lead format expected

### Signal Values Out of Range

If signals exceed ±5 mV:
- Check if image orientation is correct
- Verify ECG paper scale (standard is 10mm/mV)
- Confirm grid lines are visible

## License

MIT License - See LICENSE file
