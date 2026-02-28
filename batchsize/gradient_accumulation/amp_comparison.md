# Automatic Mixed Precision (AMP) Comparison: ON vs OFF

This document compares the gradient accumulation training results with AMP (Automatic Mixed Precision) turned ON versus OFF. Tests were conducted across various effective batch sizes (128 to 8192) using a micro batch size of 64.

## Key Findings

- **Training Speed**: Training with AMP ON is roughly **1.7x faster**. Total training time drops from ~950 seconds down to ~550 seconds.
- **VRAM Usage**: AMP ON significantly reduces peak VRAM usage by about **43%**, dropping from ~5.7 GB to ~3.2 GB.
- **Throughput**: Average samples processed per second increases drastically from ~300 samples/sec to ~540 samples/sec.
- **Model Performance**: There is no noticeable degradation in model performance (final validation accuracy and best validation loss remain basically identical).

## Detailed Metrics Comparison

| Effective Batch Size | AMP | Time (sec) | Peak VRAM (MB) | Throughput (samples/sec) | Best Val Loss | Final Val Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **128** | OFF | 963 | 5702 | 295 | 0.1000 | 0.9675 |
| | ON | 567 | 3182 | 527 | 0.1085 | 0.9644 |
| **256** | OFF | 952 | 5700 | 299 | 0.1038 | 0.9692 |
| | ON | 549 | 3179 | 541 | 0.1083 | 0.9661 |
| **512** | OFF | 945 | 5790 | 300 | 0.1072 | 0.9637 |
| | ON | 550 | 3271 | 543 | 0.1040 | 0.9658 |
| **1024** | OFF | 943 | 5700 | 301 | 0.1046 | 0.9646 |
| | ON | 549 | 3266 | 542 | 0.1063 | 0.9675 |
| **2048** | OFF | 944 | 5700 | 301 | 0.1061 | 0.9680 |
| | ON | 549 | 3266 | 543 | 0.1018 | 0.9699 |
| **4096** | OFF | 945 | 5700 | 301 | 0.1210 | 0.9654 |
| | ON | 547 | 3266 | 543 | 0.1221 | 0.9639 |
| **8192** | OFF | 943 | 5790 | 301 | 0.1391 | 0.9591 |
| | ON | 549 | 3269 | 543 | 0.1387 | 0.9591 |

## Conclusion
Using `torch.amp` (AMP ON) provides highly significant benefits for both memory and computational efficiency, making it highly recommended for future experiments involving Gradient Accumulation and these model architectures on compatible GPUs.
