from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import numpy as np
import torch
import warnings

def calculate_metrics(y_true, y_probs, y_pred):
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Calculate AUC-ROC
    try:
        if len(np.unique(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_probs[:, 1])
        else:
            metrics['auc_roc'] = float('nan')
    except Exception as e:
        metrics['auc_roc'] = float('nan')
        
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Sensitivity & Specificity
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        metrics['sensitivity'] = float('nan')
        metrics['specificity'] = float('nan')
        
    return metrics
