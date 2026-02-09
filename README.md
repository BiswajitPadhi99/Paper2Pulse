# Paper2Pulse: Robust Neural ECG Digitization and Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![iOS](https://img.shields.io/badge/iOS-16.0+-000000.svg)](https://www.apple.com/ios/)

Official implementation of **"Robust Neural ECG Digitization for Multi-Modal Interpretation Using Signals and Images"** (KDD 2026).

[📄 Paper](#) | [🎬 Demo Video](#demo-video) | [📱 iOS App](#ios-application) | [📊 Dataset](#ecg-paper-1k-dataset) | [🤗 Models](#pretrained-models)

---

## 🔬 Overview

**Paper2Pulse** is an end-to-end framework for transforming degraded paper ECG photographs into accurate diagnostic predictions. Our system bridges the gap between paper-based ECG archives—common in resource-limited healthcare settings—and modern AI-powered cardiac analysis.

<p align="center">
  <img src="assets/pipeline_overview.png" width="90%" alt="Paper2Pulse Pipeline Overview"/>
</p>

### Key Results

| Metric | Paper2Pulse | Best Baseline | Improvement |
|--------|-------------|---------------|-------------|
| **Digitization SNR** (Real Photos) | 16.41 dB | 2.06 dB | +14.35 dB |
| **Classification F1** (Real Photos) | 0.716 | 0.302 (image-based) | +137% |
| **RMSE** (Real Photos) | 0.093 mV | 0.337 mV | -72% |

### Why Digitization Matters

<p align="center">
  <img src="assets/classification_comparison.png" width="70%" alt="Classification Performance Gap"/>
</p>

Direct image-based classification collapses on real-world ECG photographs (F1=0.30), while classifying digitized signals achieves **F1=0.72**—a **2.4× improvement** that demonstrates digitization is essential, not optional.

---

## 📦 Repository Structure

```
Paper2Pulse/
├── Paper2Pulse-D/              # Digitization module
│   ├── step1_keypoint/         # Keypoint detection & perspective normalization
│   ├── step2_grid/             # Grid detection & TPS rectification
│   ├── step3_signal/           # Signal extraction with soft-argmax
│   ├── configs/                # Training configurations
│   ├── train.py                # Training script
│   ├── inference.py            # Inference pipeline
│   └── README.md               # Detailed digitization documentation
│
├── Paper2Pulse-C/              # Classification module
│   ├── models/                 # Signal encoder architecture
│   ├── configs/                # Training configurations
│   ├── train.py                # Training script
│   ├── inference.py            # Inference pipeline
│   └── README.md               # Detailed classification documentation
│
├── App/                        # iOS application
│   ├── Paper2Pulse/            # Xcode project
│   ├── CoreML/                 # Converted CoreML models
│   └── README.md               # App build instructions
│
├── data/                       # Data processing utilities
│   ├── ecg_image_kit/          # Synthetic image generation
│   ├── preprocessing/          # Data preprocessing scripts
│   └── README.md               # Dataset preparation guide
│
├── checkpoints/                # Pretrained model weights
├── assets/                     # Figures and media
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/BiswajitPadhi99/Paper2Pulse.git
cd Paper2Pulse

# Create conda environment
conda create -n paper2pulse python=3.10
conda activate paper2pulse

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### Download Pretrained Models

```bash
# Download all pretrained weights
bash scripts/download_checkpoints.sh

# Or download individually:
# Paper2Pulse-D Step 1 (Keypoint Detection)
wget -P checkpoints/ https://github.com/BiswajitPadhi99/Paper2Pulse/releases/download/v1.0/step1_keypoint.pth

# Paper2Pulse-D Step 2 (Grid Detection)
wget -P checkpoints/ https://github.com/BiswajitPadhi99/Paper2Pulse/releases/download/v1.0/step2_grid.pth

# Paper2Pulse-D Step 3 (Signal Extraction)
wget -P checkpoints/ https://github.com/BiswajitPadhi99/Paper2Pulse/releases/download/v1.0/step3_signal.pth

# Paper2Pulse-C (Classifier)
wget -P checkpoints/ https://github.com/BiswajitPadhi99/Paper2Pulse/releases/download/v1.0/classifier.pth
```

### Run Inference

**Full Pipeline (Image → Diagnosis):**

```bash
python inference.py \
    --image_path examples/sample_ecg.jpg \
    --output_dir outputs/ \
    --visualize
```

**Digitization Only:**

```bash
python Paper2Pulse-D/inference.py \
    --image_path examples/sample_ecg.jpg \
    --output_path outputs/digitized_signal.csv \
    --save_overlay
```

**Classification Only (from signal):**

```bash
python Paper2Pulse-C/inference.py \
    --signal_path outputs/digitized_signal.csv \
    --output_path outputs/predictions.json
```

### Expected Output

```
Paper2Pulse Inference Results
=============================
Input: examples/sample_ecg.jpg
Digitization SNR: 17.23 dB
Digitization RMSE: 0.087 mV

Classification Results:
  NORM: 0.12 (Normal)
  MI:   0.78 (Myocardial Infarction) ⚠️
  STTC: 0.45 (ST/T Change)
  CD:   0.08 (Conduction Disturbance)
  HYP:  0.15 (Hypertrophy)

Output saved to: outputs/
  - digitized_signal.csv
  - overlay_visualization.png
  - predictions.json
```

---

## 📊 ECG-Paper-1K Dataset

We release **ECG-Paper-1K**, a dataset of ~1,000 real-world paper ECG photographs with ground-truth signals and diagnostic labels.

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total Images | ~1,000 |
| Cardiac Conditions | 5 (NORM, MI, STTC, CD, HYP) |
| Image Source | Samsung S24 smartphone |
| Artifacts | Folds, wrinkles, handwritten annotations, varied lighting |
| Ground Truth | Digital signals + diagnostic labels |

### Artifact Distribution

<p align="center">
  <img src="assets/dataset_distribution.png" width="60%" alt="ECG-Paper-1K Distribution"/>
</p>

### Download

<!-- DATASET PLACEHOLDER -->
```bash
# Dataset download link (coming soon)
# Option 1: Direct download
wget https://example.com/ecg-paper-1k.zip

# Option 2: Hugging Face
# huggingface-cli download BiswajitPadhi99/ECG-Paper-1K

# Option 3: Kaggle
# kaggle datasets download -d biswajitpadhi99/ecg-paper-1k
```

### Data Format

```
ECG-Paper-1K/
├── images/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
├── signals/
│   ├── 00001.csv          # 12-lead signal, 500 Hz, 10 seconds
│   ├── 00002.csv
│   └── ...
├── annotations.csv         # Image ID, labels, artifact types
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### Usage Agreement

By downloading ECG-Paper-1K, you agree to:
- Use the data for research purposes only
- Not attempt to re-identify any individuals
- Cite our paper in any publications using this data

---

## 🎬 Demo Video

<!-- DEMO VIDEO PLACEHOLDER -->
<p align="center">
  <a href="https://youtube.com/watch?v=PLACEHOLDER">
    <img src="assets/demo_thumbnail.png" width="70%" alt="Demo Video Thumbnail"/>
  </a>
</p>

**[▶️ Watch Full Demo on YouTube](https://youtube.com/watch?v=PLACEHOLDER)**

The demo showcases:
- 📸 Capturing an ECG photograph with the iOS app
- ⚡ Real-time digitization and signal extraction
- 🔍 Overlay visualization for quality verification
- 📋 Classification results with confidence scores
- 📤 Signal export for external analysis

---

## 📱 iOS Application

Our fully on-device iOS app enables ECG digitization and classification without internet connectivity.

<p align="center">
  <img src="assets/ios_app_screenshots.png" width="80%" alt="iOS App Screenshots"/>
</p>

### Features

- 📷 **Camera Capture**: Photograph any standard 4×3+1 ECG printout
- ⚡ **On-Device Processing**: All computation runs locally (~39s total)
- 🔒 **Privacy-First**: No data leaves your device
- 📊 **Interactive Visualization**: 
  - Digitized 12-lead waveforms
  - Signal overlay on original image
  - Per-lead amplitude statistics
- 📤 **Export**: CSV signal export for external tools

### System Requirements

- iOS 16.0+
- iPhone 12 or newer (A14 Bionic chip or later)
- ~500 MB storage for models

### Installation

<!-- APP PLACEHOLDER -->
```bash
# Option 1: TestFlight (Beta)
# Join beta: https://testflight.apple.com/join/PLACEHOLDER

# Option 2: Build from source
cd App/
open Paper2Pulse.xcodeproj
# Configure signing in Xcode, then build to device
```

### Performance Metrics (iPhone 16 Pro)

| Stage | Latency | Memory |
|-------|---------|--------|
| Step 1: Keypoint Detection | 13.0s | - |
| Step 2: Grid Detection | 24.6s | - |
| Step 3: Signal Extraction | 1.8s | - |
| Classification | 0.04s | - |
| **Total** | **39.4s** | **2.2 GB peak** |

---

## 🏋️ Training

### Data Preparation

1. **Download source datasets:**
   ```bash
   # PTB-XL
   wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
   
   # CODE-15%
   # Download from: https://zenodo.org/record/4916206
   ```

2. **Generate synthetic training images:**
   ```bash
   python data/generate_synthetic.py \
       --source_dir data/ptb-xl/ \
       --output_dir data/synthetic/ \
       --noise_levels low medium high \
       --num_workers 8
   ```

3. **Prepare data splits:**
   ```bash
   python data/prepare_splits.py \
       --data_dir data/synthetic/ \
       --output_dir data/splits/
   ```

### Train Paper2Pulse-D (Digitization)

```bash
# Step 1: Keypoint Detection
python Paper2Pulse-D/train.py \
    --config Paper2Pulse-D/configs/step1_keypoint.yaml \
    --data_dir data/synthetic/ \
    --output_dir experiments/step1/

# Step 2: Grid Detection
python Paper2Pulse-D/train.py \
    --config Paper2Pulse-D/configs/step2_grid.yaml \
    --data_dir data/synthetic/ \
    --output_dir experiments/step2/

# Step 3: Signal Extraction
python Paper2Pulse-D/train.py \
    --config Paper2Pulse-D/configs/step3_signal.yaml \
    --data_dir data/synthetic/ \
    --output_dir experiments/step3/
```

### Train Paper2Pulse-C (Classification)

```bash
python Paper2Pulse-C/train.py \
    --config Paper2Pulse-C/configs/classifier.yaml \
    --data_dir data/ptb-xl/ \
    --output_dir experiments/classifier/
```

### Training Configuration

Key hyperparameters (see `configs/` for full details):

| Parameter | Step 1 | Step 2 | Step 3 | Classifier |
|-----------|--------|--------|--------|------------|
| Encoder | ResNet-18 | ResNet-34 | EfficientNet-B3 | Custom |
| Input Size | 512×640 | 1152×1440 | 256×1280 | 5000 samples |
| Batch Size | 16 | 4 | 8 | 32 |
| Learning Rate | 1e-4 | 1e-4 | 1e-4 | 5e-4 |
| Epochs | 100 | 100 | 100 | 100 |

---

## 📈 Evaluation

### Reproduce Paper Results

```bash
# Digitization evaluation
python evaluate_digitization.py \
    --model_dir checkpoints/ \
    --data_dir data/ecg-paper-1k/ \
    --output_dir results/digitization/

# Classification evaluation
python evaluate_classification.py \
    --model_path checkpoints/classifier.pth \
    --data_dir data/ecg-paper-1k/ \
    --output_dir results/classification/

# Full pipeline evaluation
python evaluate_pipeline.py \
    --model_dir checkpoints/ \
    --data_dir data/ecg-paper-1k/ \
    --output_dir results/pipeline/
```

### Expected Results

**Digitization Performance:**

| Dataset | Method | SNR (dB) ↑ | RMSE (mV) ↓ |
|---------|--------|------------|-------------|
| Synthetic (Low Noise) | ECGMiner | 16.82 | 0.045 |
| | Krones et al. | 17.43 | 0.041 |
| | **Paper2Pulse-D** | **24.17** | **0.021** |
| Synthetic (High Noise) | ECGMiner | 8.74 | 0.204 |
| | Krones et al. | 13.38 | 0.094 |
| | **Paper2Pulse-D** | **15.22** | **0.082** |
| ECG-Paper-1K (Real) | ECGMiner | -4.71 | 0.613 |
| | Krones et al. | 2.06 | 0.337 |
| | **Paper2Pulse-D** | **16.41** | **0.093** |

**Classification Performance (ECG-Paper-1K):**

| Modality | Method | F1 ↑ | AUC ↑ |
|----------|--------|------|-------|
| Image | ResNet-50 | 0.128 | 0.526 |
| | ECGConvNext | 0.237 | 0.498 |
| | PULSE | 0.239 | 0.517 |
| Digitized Signal | Zhang et al. | 0.642 | 0.758 |
| | ST-MEM | 0.701 | 0.806 |
| | ECG-FM | 0.689 | 0.787 |
| | **Paper2Pulse-C** | **0.716** | **0.824** |

---

## 🛠️ Model Architecture

### Paper2Pulse-D: Three-Step Digitization Pipeline

<p align="center">
  <img src="assets/digitization_architecture.png" width="95%" alt="Digitization Architecture"/>
</p>

1. **Step 1 - Keypoint Detection**: ResNet-18 encoder + U-Net decoder detecting 9 lead label keypoints for RANSAC-based homography estimation

2. **Step 2 - Grid Detection**: ResNet-34 encoder + Attention U-Net decoder localizing 2,508 grid intersections for TPS rectification

3. **Step 3 - Signal Extraction**: EfficientNet-B3 encoder + coordinate-augmented decoder with differentiable soft-argmax for sub-pixel signal recovery

### Paper2Pulse-C: Hybrid CNN-Transformer Classifier

```
Input: 12-lead signal (500 Hz × 10s)
    ↓
Multi-Scale 1D Convolutions (k ∈ {3, 7, 15, 31})
    ↓
Residual Projection + Feature Concatenation
    ↓
Tokenization + Positional/Lead/Offset Embeddings
    ↓
6× Transformer Blocks (8 heads, d=256)
    ↓
MLP Classification Head → 5 diagnostic classes
```

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{padhi2026paper2pulse,
  title={Robust Neural ECG Digitization for Multi-Modal Interpretation Using Signals and Images},
  author={Padhi, Biswajit and Yin, Changchang and Liu, Ruoqi and Wang, Pengqi and Zhang, Ping},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026},
  organization={ACM}
}
```

---

## 🙏 Acknowledgments

- [PTB-XL](https://physionet.org/content/ptb-xl/) for the ECG signal database
- [CODE-15%](https://zenodo.org/record/4916206) for additional ECG records
- [ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit) for synthetic image generation
- [PhysioNet 2024 Challenge](https://moody-challenge.physionet.org/2024/) for evaluation datasets

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

- **Biswajit Padhi** - padhi.3@osu.edu
- **Ping Zhang** - zhang.10631@osu.edu

For questions about the paper, code, or dataset, please open an issue or contact us directly.

---

<p align="center">
  <b>⭐ If you find Paper2Pulse useful, please consider giving us a star! ⭐</b>
</p>
