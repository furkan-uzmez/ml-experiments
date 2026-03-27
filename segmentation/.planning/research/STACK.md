# Stack Research — Medical Image Segmentation Benchmark

## Recommended Stack (2025)

### Core Framework
- **PyTorch ≥ 2.2** — Primary deep learning framework; all three models (U-Net, nnU-Net, MedSAM) use PyTorch
- **CUDA ≥ 12.1** — Required for modern GPU support and mixed-precision training
- **Python 3.10–3.12** — Best compatibility across all dependencies

### Model-Specific Dependencies

| Component | Library | Version | Rationale |
|-----------|---------|---------|-----------|
| nnU-Net | `nnunetv2` | ≥ 2.5 | Latest self-configuring framework with automated preprocessing, training, and inference |
| MedSAM | `segment-anything` + MedSAM checkpoint | Latest | Foundation model adapted for medical imaging; requires ViT-B image encoder |
| U-Net | Custom or `monai` | ≥ 1.3 | MONAI provides validated medical U-Net implementations with proper normalization |

### Medical Imaging I/O
- **SimpleITK ≥ 2.3** — Reading/writing NIfTI, DICOM with metadata preservation
- **nibabel ≥ 5.2** — NIfTI file handling (complementary to SimpleITK)
- **pydicom ≥ 2.4** — DICOM parsing if dataset is in DICOM format

### Evaluation & Metrics
- **surface-distance** (DeepMind) — HD95 and surface-based metrics
- **medpy ≥ 0.5** — Medical image metric computation (Dice, HD95)
- **scipy ≥ 1.12** — Distance transforms for Hausdorff computation
- **numpy ≥ 1.26** — Array operations

### Visualization & Reporting
- **matplotlib ≥ 3.8** — Boxplots, bar charts, comparison figures
- **seaborn ≥ 0.13** — Statistical visualization
- **pandas ≥ 2.1** — Result aggregation and CSV generation

### Key Libraries NOT to Use
- **TensorFlow** — All three models are PyTorch-native; mixing frameworks adds complexity
- **OpenCV for medical I/O** — Use SimpleITK instead; OpenCV doesn't handle 3D medical formats properly
- **Custom metric implementations** — Use validated libraries (medpy, surface-distance) to avoid bugs

## Confidence Levels

| Recommendation | Confidence | Notes |
|---------------|------------|-------|
| PyTorch as framework | ★★★★★ | All three models are PyTorch-native |
| nnU-Net v2 | ★★★★★ | Standard automated baseline, actively maintained |
| MONAI for U-Net | ★★★★☆ | Well-validated but custom U-Net is also acceptable |
| SimpleITK for I/O | ★★★★★ | Standard in medical imaging pipelines |
| medpy for metrics | ★★★★☆ | Validated, but check HD95 implementation consistency |
| MedSAM checkpoint | ★★★★☆ | Active development; verify checkpoint version |
