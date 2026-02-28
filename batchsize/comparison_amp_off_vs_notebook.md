# Comparison: Standard Batch Size vs. Gradient Accumulation (AMP Off)

This report compares the ResNet50 training results from standard batch sizing ([batch_size_experiment_notebook.ipynb](file:///d:/Projects/ML_Algorithms/batchsize/batch_size_experiment_notebook.ipynb)) to the gradient accumulation methodology (`gradient_accumulation/amp_off` runs), using the `COVIDCXNetDataset`.

## 1. Maximum Batch Size and VRAM Usage

One of the largest limitations of standard training is the GPU Memory (VRAM) bottleneck. As batch sizes increase, memory usage scales linearly. Gradient accumulation bypasses this by utilizing a fixed "micro batch size" (MB) and accumulating gradients over multiple steps to simulate a large "effective batch size" (EBS).

### Holistic Comparison: VRAM, Validation Accuracy, and Training Time

* **Model:** ResNet50
* **Standard Training Limit:** Max 256 (Uses ~21.9GB VRAM). At 512+, `CUDA Out Of Memory (OOM)`.
* **Gradient Accumulation:** Uses a fixed Micro Batch Size of 64. Memory remains constant at ~5.7 GB regardless of Effective Batch Size, with no significant time penalty.

| Batch Size / EBS | Method | Acc. Steps | Peak VRAM (MB) | Best Val Acc | Total Time (s) | Status |
|---|---|---|---|---|---|---|
| 8 | Standard | - | 1004.54 | 0.9654 | N/A | Success |
| 16 | Standard | - | 2027.31 | 0.9651 | N/A | Success |
| 32 | Standard | - | 3339.70 | 0.9649 | N/A | Success |
| 64 | Standard | - | 5988.86 | 0.9668 | N/A | Success |
| 128 | Standard | - | 11288.27 | 0.9670 | N/A | Success |
| 128 | Grad. Acc. | 2 | 5701.72 | 0.9675 | 962.9 | Success |
| 211 | Standard | - | 18159.35 | 0.9630 | N/A | Success |
| 256 | Standard | - | 21943.18 | N/A | N/A | Success |
| 256 | Grad. Acc. | 4 | 5700.47 | 0.9691 | 951.7 | Success |
| 512+ | Standard | - | OOM | N/A | N/A | Failed |
| 512 | Grad. Acc. | 8 | 5789.90 | 0.9636 | 944.8 | Success |
| 1024 | Grad. Acc. | 16 | 5700.47 | 0.9646 | 943.4 | Success |
| 2048 | Grad. Acc. | 32 | 5700.47 | 0.9679 | 943.7 | Success |
| 4096 | Grad. Acc. | 64 | 5700.47 | 0.9653 | 945.4 | Success |
| 8192 | Grad. Acc. | 128 | 5789.90 | 0.9591 | 942.8 | Success |

> **Conclusion**: Standard training runs out of memory on batch sizes above 256. Gradient accumulation is highly successful in decoupling VRAM usage from the targeted batch size, stably allowing an effective batch size of **8192** while consuming less than 6GB of VRAM. It preserves the validation accuracy dynamics well without incurring a runtime penalty when holding the micro-batch size constant.

---

## 3. Training Time Dynamics

The runtime per epoch remains highly consistent on Gradient Accumulation across all EBS tiers because the total number of forward/backward passes evaluated over the entire dataset remains identical.

| Effective Batch Size | Total Train Time (sec) | Avg Samples Per Sec |
|---|---|---|
| 128 | 962.9 | 295.4 |
| 256 | 951.7 | 298.7 |
| 512 | 944.8 | 300.3 |
| 1024 | 943.4 | 301.3 |
| 2048 | 943.7 | 300.8 |
| 4096 | 945.4 | 300.8 |
| 8192 | 942.8 | 301.4 |

> **Note**: Gradient accumulation ensures seamless scaling without incurring a runtime penalty when holding the micro-batch size constant, maintaining steady throughput around ~300 samples/sec on the target hardware.
