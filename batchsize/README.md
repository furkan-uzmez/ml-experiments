# Batch Size Experiments Overview

This repository folder contains the code and notebooks used to experiment with and determine the optimal batch size for training image classification models (specifically ResNet-18, ResNet-34, and ResNet-50) on the COVIDCXNet dataset.

## What Was Done
1. **Max Batch Size Search**: We implemented a script (`find_max_batch.py`) to systematically search for the absolute maximum batch size the GPU can handle before running out of memory (OOM). It uses a binary search algorithm to test batch sizes.
2. **Sample Experiment for Batch Sizes**: We created a function (`sample_experiment.py`) to test standard power-of-two batch sizes (e.g., 8, 16, 32... 8192) on a subset of the data (or the full dataset) and measure the peak memory usage for each. We recently updated this script to support fractional dataset splits (e.g., using `0.5` to test on 50% of the data).
3. **Execution & Analysis in Notebook**: The `batch_size_experiment_notebook.ipynb` ties these tools together. It loads the dataset, executes the binary search to find the theoretical limits, and then runs the standard batch sizes to record their actual memory footprints and verify which ones succeed or hit OOM errors.

## Code and Notebook Analysis
- **`find_max_batch_size(device, model_name='resnet18')`**: Found that for a ResNet-18 model on this hardware (CUDA), the maximum absolute batch size is **1049**. Based on this, it recommends a safe batch size of **839** (which is 80% of the absolute max) to account for memory spikes during backpropagation in full training.
- **`find_max_batch_size(device, model_name='resnet34')`**: The maximum absolute batch size for ResNet-34 is **720**, with a recommended safe batch size of **576**.
- **`find_max_batch_size(device, model_name='resnet50')`**: The maximum absolute batch size for ResNet-50 is **270**, with a recommended safe batch size of **216**.
- **`run_sample_experiment(...)`**: Tested power-of-two batch sizes up to 8192.
  - As batch size doubles, the memory required scales somewhat linearly but is dominated by activations and gradients.
  - **ResNet-18**: Sizes from 8 up to 1024 succeeded without error. At batch size 1024, the peak memory reached `~22,321 MB`.
  - **ResNet-34**: Sizes from 8 up to 512 succeeded. At batch size 512, peak memory was `~20,830 MB`.
  - **ResNet-50**: Sizes from 8 up to 256 succeeded. At batch size 256, peak memory was `~21,880 MB`.
  - Batch sizes exceeding these limits predictability failed with Out Of Memory (OOM) errors.
  - **Key Observation:** The maximum absolute batch size significantly decreases as the model architecture becomes larger and more complex (1049 for ResNet-18 -> 720 for ResNet-34 -> 270 for ResNet-50). This occurs because deeper models contain more parameters and intermediate activations, which consume significantly more VRAM per sample during training on the same dataset.

## Conclusion and Results
The experiments clearly define the hardware boundaries for training ResNet models on this setup.

### ResNet-18
- **Absolute Maximum limit**: 1049
- **Largest usable standard power-of-two batch size**: 1024 (uses ~22.3GB VRAM)
- **Safe Recommended Custom Batch Size**: ~839

### ResNet-34
- **Absolute Maximum limit**: 720
- **Largest usable standard power-of-two batch size**: 512 (uses ~20.8GB VRAM)
- **Safe Recommended Custom Batch Size**: ~576

### ResNet-50
- **Absolute Maximum limit**: 270
- **Largest usable standard power-of-two batch size**: 256 (uses ~21.9GB VRAM)
- **Safe Recommended Custom Batch Size**: ~216

**Takeaway:** For stable and fast training without risking OOM crashes midway through an epoch, select a batch size below the absolute maximum for your architecture. If you prefer standard power-of-two sizes, choose a size that fits comfortably within the 24GB VRAM limit (e.g., 1024 for ResNet-18, 512 for ResNet-34, 256 for ResNet-50). If running alongside other processes, the recommended safe capacity estimates serve as reliable targets.

### Generalization Gap Consideration
While identifying the maximum batch size prevents Out Of Memory (OOM) errors and maximizes hardware efficiency, it is critical to keep the **"Generalization Gap"** in mind. It is a well-documented phenomenon in deep learning that models trained with extremely large batch sizes often suffer a degradation in generalization performance (resulting in lower test/validation accuracy). 
- Smaller batch sizes introduce noise into the gradient estimates, which helps the model escape "sharp" local minima and settle into "flat" minima that generalize better to unseen data.
- Very large batch sizes provide highly accurate gradient estimates but tend to converge into sharper minima, losing out on generalization power.
Therefore, the *optimal* batch size is a continuous trade-off between **training speed / memory utilization** and **model accuracy**. Even if your GPU can support a batch size of 1024, a smaller size like 64 or 128 might theoretically yield a model with higher test metrics.

**Empirical Findings in this Project:** Interestingly, in our specific experiments with ResNet-34 and ResNet-50, we did not observe a significant or consistent generalization gap. Validation accuracy remained relatively stable across the successfully tested batch sizes, and the expected severe degradation at larger batch regimes was not strongly evident in our results. This indicates that for this dataset and model setup, maximizing batch size for training speed might not heavily penalize model performance, but this may happen because of we couldnt try higher batch sizes like 2048, 4096, 8192, 16384 due to lack of vram.
