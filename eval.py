import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

threshold = 0.5

def threat_score(y_true, y_pred):
    """Threat Score = TP / (TP + FN + FP)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    if (tp + fn + fp) == 0:
        return 0.0
    return tp / (tp + fn + fp)

def threat_score_loss(probs, targets, epsilon=1e-6):
    """
    Calculates the negative of the differentiable Threat Score (Jaccard Index/IoU).
    Outputs are logits, targets are 0 or 1.
    """
    targets = targets.float() # Ensure targets are float
        
    # 2. Calculate differentiable components
    TP_approx = torch.sum(targets * probs)
    FN_approx = torch.sum(targets * (1 - probs))
    FP_approx = torch.sum((1 - targets) * probs)
    
    # 3. Threat Score (TS) calculation
    denominator = TP_approx + FN_approx + FP_approx + epsilon
    threat_score = TP_approx / denominator
    
    # 4. Loss is the negative of the metric
    loss = -threat_score
    return loss

class ThreatScoreLoss(nn.Module):
    """
    A differentiable approximation of the 1 - Threat Score (CSI) loss.
    This loss is equivalent to a variant of the Dice/F-beta loss.
    
    It focuses on minimizing False Negatives (FN) and False Positives (FP)
    while maximizing True Positives (TP).
    """
    def __init__(self, epsilon=1e-6):
        super(ThreatScoreLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Flatten tensors for simpler calculation
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1).float() # Ensure targets are float (0.0 or 1.0)

        # 1. True Positives (TP) approximation:
        # y_pred * y_true -> High when both are 1.0. This is the intersection.
        intersection = (y_pred * y_true).sum()

        # 2. False Positives (FP) approximation:
        # y_pred * (1 - y_true) -> High when prediction is 1.0 and true is 0.0.
        fp = (y_pred * (1 - y_true)).sum()

        # 3. False Negatives (FN) approximation:
        # (1 - y_pred) * y_true -> High when prediction is 0.0 and true is 1.0.
        fn = ((1 - y_pred) * y_true).sum()
        
        # --- TS Differentiable Approximation (Dice-like) ---
        # The true TS denominator is (TP + FP + FN)
        # We use a similar structure that is differentiable:
        
        numerator = intersection 
        
        # Denominator: TP + FP + FN. 
        # This is equivalent to: (y_pred * y_true) + (y_pred * (1-y_true)) + ((1-y_pred) * y_true)
        # which simplifies mathematically to: y_pred + y_true - (y_pred * y_true)
        denominator = numerator + fp + fn
        
        # The metric to maximize (Threat Score approximation):
        # ts_approx = numerator / denominator
        ts_approx = (numerator + self.epsilon) / (denominator + self.epsilon)
        
        # The loss to minimize: 1 - ts_approx
        loss = 1.0 - ts_approx

        return loss

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    model.eval()
    with torch.no_grad():
        y_pred, _ = model(X_test, None)
        y_pred = y_pred.cpu().numpy()
    
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_test = y_test.cpu().numpy()

    ts = threat_score(y_test, y_pred_binary)
    jaccard = jaccard_score(y_test, y_pred_binary, pos_label=1)
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary, labels=[0, 1]).ravel()
    #print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    return {
        'name': model.name,
        'ts': ts,
        'jaccard': jaccard,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }

def compare_params(models, durations):
    """Compare learned vs published Staley 2017 parameters"""
    published = {
        '15min': {'B': -3.63, 'Ct': 0.41, 'Cf': 0.67, 'Cs': 0.70},
        '30min': {'B': -3.61, 'Ct': 0.26, 'Cf': 0.39, 'Cs': 0.50},
        '60min': {'B': -3.21, 'Ct': 0.17, 'Cf': 0.20, 'Cs': 0.22}
    }
    
    print(f"\n{'='*60}")
    print("Comparison with Staley 2017 Published Parameters")
    print(f"{'='*60}")
    
    for dur in durations:
        print(f"\n{dur}:")
        print(f"{'Param':<8} {'Published':<12} {'Learned':<12} {'Diff':<12}")
        print("-" * 50)
        
        for param in ['B', 'Ct', 'Cf', 'Cs']:
            pub = published[dur][param]
            learn = models[dur].state_dict()[param].item()
            diff = learn - pub
            print(f"{param:<8} {pub:<12.4f} {learn:<12.4f} {diff:<12.4f}")