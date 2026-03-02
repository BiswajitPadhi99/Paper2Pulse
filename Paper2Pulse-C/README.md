# Paper2Pulse-C

Hybrid CNN-Transformer classifier for cardiac abnormality diagnosis from digitized 12-lead ECG signals.

## Files

| File | Description |
|------|-------------|
| `model.py` | Paper2Pulse-C architecture |
| `losses.py` | Adaptive multi-label loss (asymmetric + focal + dice) |
| `dataset.py` | PTB-XL and CODE-15% data loaders |
| `train.py` | Single split training |
| `train_cv.py` | Stratified k-fold cross-validation training |
| `metrics.py` | Macro F1, AUROC, threshold optimization |
| `inference.py` | Inference on digitized signals |
| `augmentations.py` | Signal-level augmentations |
| `experiment_runner.py` | Single experiment runner |
| `run_experiments.py` | Augmentation ablation orchestrator |

## Training

```bash
# Cross-validation (recommended)
python train_cv.py --data_dir /path/to/data/folder --n_folds 10 --epochs 100

# Single split
python train.py --data_dir /path/to/data/folder --epochs 100
```

## Inference

```bash
python inference.py --checkpoint best_model.pt --signal signal.npy
```
