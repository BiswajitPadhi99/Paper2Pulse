# Paper2Pulse-D

Three-step neural ECG digitization pipeline: keypoint detection → grid detection → signal extraction.

## Pipeline

**Step 1 — Keypoint Detection:** Detects 9 lead-label landmarks (aVR, V1, V4, aVL, V2, V5, aVF, V3, V6) via heatmap regression. Computes a RANSAC homography to correct perspective distortion.

**Step 2 — Grid Detection:** Jointly segments all grid intersection points and classifies each to its horizontal/vertical line index. Thin-Plate Spline (TPS) warping corrects local deformations from paper warping or lens distortion.

**Step 3 — Signal Extraction:** Per-lead segmentation with differentiable soft-argmax produces continuous, calibrated waveform amplitudes. A composite loss combines pixel-level focal loss with signal-level wing loss and derivative losses to preserve waveform morphology.

---
*Code and model weights will be updated soon.*