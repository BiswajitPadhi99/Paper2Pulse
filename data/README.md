# ECG-Paper-1K

Dataset of 1,000 smartphone photographs of printed ECGs with paired ground-truth signals and diagnostic labels.

**50-sample preview available for review. Full dataset released upon acceptance.**

## Construction

ECG records from PTB-XL [1], printed on standard letter-sized paper, and photographed using a Samsung S24 smartphone (12 MP) under varied lighting conditions and angles. Controlled real-world artifacts were physically introduced prior to capture.

## Format

| Field | Description |
|-------|-------------|
| `image_filename` | JPEG photograph, available in `\data\images`|
| `record` | Source PTB-XL record ID, available in `\data\signals` folder |
| `labels` | Multi-label diagnosis (NORM, MI, STTC, CD, HYP) |
| `rotation` | Rotation artifact present (0/1) |
| `fold` | Paper fold artifact present (0/1) |
| `wrinkle` | Wrinkle artifact present (0/1) |
| `hw_text` | Handwritten annotation present (0/1) |
| `device` | Camera device |

## Artifact Distribution

| Artifact | Count |
|----------|-------|
| Rotation (R) | 711 |
| Folding (F) | 540 |
| Wrinkles (W) | 341 |
| Handwritten text (HW) | 162 |

## References

[1] Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset," *Scientific Data*, 2020.