import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

threshold = 0.5

def threat_score(y_true, y_pred):
    """Threat Score = TP / (TP + FN + FP)"""
    y_pred_binary = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    
    if (tp + fn + fp) == 0:
        return 0.0
    return tp / (tp + fn + fp)

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().flatten()
    
    y_pred_binary = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test.cpu().numpy(), y_pred_binary, labels=[0, 1]).ravel()
    
    ts = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{'='*50}")
    print(f"Test Results")
    print(f"Threat Score: {ts:.4f}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print(f"Threshold:    {threshold:.3f}")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    return ts

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

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
    
    y_test = y_test.cpu().numpy()

    y_pred_binary = (y_pred >= threshold).astype(int)
    
    #print(y_pred_binary)

    ts = threat_score(y_test, y_pred_binary)
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary, labels=[0, 1]).ravel()
    #print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    return {
        'name': model.name,
        'ts': ts,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }