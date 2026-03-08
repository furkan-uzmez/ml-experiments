# Random Seed Comparison Experiment Results

This document summarizes the training and testing results for a model trained with three different random seeds (`42`, `123`, and `2024`). The goal is to observe the impact of random seed initialization on model performance.

## Training Metrics

| Seed | Best Epoch | Best Val Loss | Best Val Accuracy | Training Stopped At | Early Stopping |
|---|---|---|---|---|---|
| **42** | 3 | 0.0965 | 0.9689 | Epoch 5 | Counter: 2/3 |
| **123** | 2 | 0.0906 | 0.9675 | Epoch 5 | Triggered (3/3) |
| **2024** | 2 | **0.0860** | 0.9682 | Epoch 5 | Triggered (3/3) |

*Note: All seeds trained effectively, achieving top validation accuracies around 96.7% - 96.8%. Seed 2024 achieved the lowest validation loss (0.0860) at epoch 2. Both seeds 123 and 2024 triggered early stopping at epoch 5.*

## Test Set Performance

| Seed | Test Accuracy | Test F1-Score |
|---|---|---|
| **42** | 0.9649 | 0.9622 |
| **123** | 0.9721 | 0.9706 |
| **2024** | **0.9744** | **0.9730** |

## Conclusion and Observations

- **Performance Variation:** There is a noticeable difference in final test performance based purely on the random seed. Test accuracy varied from ~96.5% to ~97.4%, and the F1-score varied from ~96.2% to ~97.3%.
- **Best Seed:** **Seed 2024** yielded the best overall generalization results on the unseen test set, providing the highest accuracy (0.9744) and F1-score (0.9730). It also achieved the lowest validation loss during training.
- **Consistency:** Seed 42, which achieved the highest validation accuracy (0.9689) among the three during training, ended up having the lowest performance on the test set. This highlights the importance of evaluating on a thoroughly held-out test set, as validation performance can sometimes be slightly misleading depending on the subset and seed variance.
