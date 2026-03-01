import numpy as np
from sklearn import metrics

def calculate_extended_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate extended metrics for deepfake detection
    
    Args:
        y_true: Ground truth labels (0=real, 1=fake)
        y_pred_proba: Predicted probabilities for fake class
        threshold: Classification threshold (default 0.5)
    
    Returns:
        dict: Extended metrics
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    
    # Basic metrics
    accuracy = metrics.accuracy_score(y_true, y_pred)
    
    # Precision & Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Also called TPR or Sensitivity
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # False Positive Rate
    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # False Negative Rate
    fnr_value = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # ROC AUC
    try:
        auc = metrics.roc_auc_score(y_true, y_pred_proba)
    except:
        auc = 0.0
    
    # Average Precision (AP)
    try:
        ap = metrics.average_precision_score(y_true, y_pred_proba)
    except:
        ap = 0.0
    
    # EER calculation
    try:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_proba, pos_label=1)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    except:
        eer = 0.0
    
    return {
        # Basic
        'accuracy': accuracy,
        'auc': auc,
        'ap': ap,
        'eer': eer,
        
        # Extended
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'fpr': fpr_value,
        'fnr': fnr_value,
        
        # Confusion matrix
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        
        # Threshold used
        'threshold': threshold
    }


def print_metrics_summary(metrics_dict):
    """
    Print formatted metrics summary
    """
    print("\n" + "="*70)
    print("                      VYSLEDKY TESTU")
    print("="*70)
    
    print("\n--- HLAVNE METRIKY ---")
    print(f"  AUC (Area Under Curve):     {metrics_dict['auc']:.4f}")
    print(f"  Accuracy (Presnost):        {metrics_dict['accuracy']:.4f} ({metrics_dict['accuracy']*100:.2f}%)")
    print(f"  EER (Equal Error Rate):     {metrics_dict['eer']:.4f}")
    print(f"  AP (Average Precision):     {metrics_dict['ap']:.4f}")
    
    print("\n--- DOPLNKOVE METRIKY ---")
    print(f"  Precision (Fake):           {metrics_dict['precision']:.4f}")
    print(f"  Recall/TPR (Fake):          {metrics_dict['recall']:.4f}")
    print(f"  F1-Score:                   {metrics_dict['f1_score']:.4f}")
    print(f"  Specificity/TNR (Real):     {metrics_dict['specificity']:.4f}")
    print(f"  FPR (False Positive Rate):  {metrics_dict['fpr']:.4f}")
    print(f"  FNR (False Negative Rate):  {metrics_dict['fnr']:.4f}")
    
    print("\n--- CONFUSION MATRIX ---")
    print(f"  True Positives (Fake):      {metrics_dict['true_positive']}")
    print(f"  True Negatives (Real):      {metrics_dict['true_negative']}")
    print(f"  False Positives:            {metrics_dict['false_positive']}")
    print(f"  False Negatives:            {metrics_dict['false_negative']}")
    
    print("\n" + "="*70)
    
    # Interpretation
    print("\nINTERPRETACIA:")
    if metrics_dict['auc'] >= 0.9:
        print("  \u2713 AUC >= 0.9: VYBORNE! Model velmi dobre rozoznava fake od real.")
    elif metrics_dict['auc'] >= 0.8:
        print("  \u2713 AUC >= 0.8: DOBRE. Model rozoznava fake od real.")
    elif metrics_dict['auc'] >= 0.7:
        print("  ~ AUC >= 0.7: PRIJATELNE. Model ma problemy s niektirymi fake.")
    elif metrics_dict['auc'] >= 0.6:
        print("  ! AUC >= 0.6: SLABE. Model ma vela chyb.")
    else:
        print("  X AUC < 0.6: ZLE! Model je takmer ako nahodne hadanie.")
    
    print("="*70 + "\n")

