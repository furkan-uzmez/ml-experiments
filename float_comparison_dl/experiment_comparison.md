# Precision Comparison Results (FP32 vs FP16 vs AMP)

Based on the experiment results running on medical image classification for 5 epochs with a ResNet50 model, here is the comparison between standard 32-bit floating point (FP32), 16-bit floating point (FP16), and Automatic Mixed Precision (AMP).

## Performance Metrics Comparison

| Precision | Avg Time/Epoch (s) | Total Time (s) | Peak Memory (MB) | Final Val Acc | Final Val AUC | Final Val F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 225.07 | 1125.36 | 2955.80 | 0.9701 | 0.9940 | 0.9651 |
| **FP16** | 141.99 | 709.94 | 1729.32 | 0.6954 | NaN | 0.4102 |
| **AMP** | 140.31 | 701.55 | 2003.46 | 0.9721 | 0.9935 | 0.9672 |

## Key Findings

1. **Training Speed & Efficiency**
   - **FP32** is the slowest, taking ~225 seconds per epoch.
   - Both **FP16** and **AMP** provide significant speedups (nearly 38% faster), reducing the average epoch time to roughly ~141 seconds.
   
2. **Memory Usage (VRAM)**
   - **FP32** consumes the most memory at ~2.96 GB.
   - **FP16** uses the least memory (~1.73 GB), a reduction of roughly 41%.
   - **AMP** sits in between but closer to FP16, using ~2.00 GB (32% less memory than FP32).

3. **Model Accuracy & Stability**
   - **FP32** provides strong, stable performance as expected (AUC: 0.9940).
   - **FP16 (Pure)** failed to converge correctly. It experienced numerical instability (underflow/overflow), resulting in `NaN` losses and AUC starting from the first epoch. Its accuracy plummeted to 69.5%, rendering the model ineffective.
   - **AMP** perfectly stabilizes the training while retaining the speed and memory benefits. It achieves an accuracy (0.9721) and AUC (0.9935) virtually identical to FP32.

## Conclusion

**Automatic Mixed Precision (AMP) is the clear winner.** It successfully captures the computational speedups and memory reductions of FP16 without suffering from the numerical instability that breaks pure FP16 training. By dynamically scaling gradients, AMP maintains the high accuracy of FP32 while saving ~32% VRAM and training ~38% faster.
