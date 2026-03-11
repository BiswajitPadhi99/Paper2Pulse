# Paper2Pulse: Robust Neural ECG Digitization and Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![iOS](https://img.shields.io/badge/iOS-16.0+-000000.svg)](https://www.apple.com/ios/)

Official implementation of **"Paper2Pulse: An End-to-End Framework for Paper ECG Digitization, Diagnosis, and On-Device Deployment"** (KDD 2026).

## Overview

**Paper2Pulse** transforms degraded paper ECG photographs into accurate diagnostic predictions. The system addresses a critical gap in resource-limited healthcare settings where ECG data exists only as paper printouts photographed on mobile phones, inaccessible to automated diagnostic systems.


## 🎬 Demo Video

<!-- DEMO VIDEO -->
<p align="center">
  <video src="img/ECG_Analyzer_Demo_Vid.mov" width="40%" controls="controls"></video>
</p>
The demo showcases:
- 📸 Capturing an ECG photograph with the iOS app
- ⚡ Real-time digitization and signal extraction
- 🔍 Overlay visualization for quality verification
- 📋 Classification results with confidence scores
- 📤 Signal export for external analysis

## Installation

### Clone the Repository

```bash
# Install Git LFS first (required for model files)
brew install git-lfs
git lfs install

# Clone the repository
git clone https://github.com/BiswajitPadhi99/Paper2Pulse.git
cd Paper2Pulse
git lfs pull
```
Next, refer the README.md in each folder for their task specific instructions.

## Repository Structure

```
Paper2Pulse/
├── Paper2Pulse-D/          # Digitization pipeline
├── Paper2Pulse-C/          # Classification model
├── data/                   # ECG-Paper-1K (50-sample preview)
└── app/                    # iOS application
```

## Components

| Component | Description |
|-----------|-------------|
| **Paper2Pulse-D** | Three-step digitization: keypoint detection → grid localization → signal extraction |
| **Paper2Pulse-C** | Hybrid CNN-Transformer classifier for digitized 12-lead signals |
| **ECG-Paper-1K** | 1,000 smartphone photographs with ground-truth signals and diagnostic labels |
| **iOS App** | Fully on-device pipeline — no internet required |

## Citation

```bibtex
@inproceedings{padhi2026paper2pulse,
  title     = {Paper2Pulse: An End-to-End Framework for Paper ECG Digitization, Diagnosis, and On-Device Deployment},
  author    = {Padhi, Biswajit and Yin, Changchang and Wang, Pengqi and Cao, Weidan and Zhang, Ping},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026}
}
```
---