# -*- coding: utf-8 -*-
"""
================================================================================
DEEPFAKE DETECTOR EVALUATION SCRIPT (Xception)
================================================================================
Author: Diploma Thesis
Description: Evaluation of Xception deepfake detector on custom dataset
             with per-image CSV logging, checkpoint support, and metrics.

Usage:
  python evaluate_detector.py --detector_path <config.yaml> \
                              --weights_path <model.pth> \
                              --test_dataset <dataset_name> \
                              [--max_samples N] [--resume]
================================================================================
"""

import os
import sys
import csv
import json
import pickle
import argparse
import numpy as np
import yaml
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, average_precision_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add training directory to PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TRAINING_DIR = os.path.join(PROJECT_ROOT, 'training')
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, PROJECT_ROOT)

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
#  ARGUMENT PARSING
# ==============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Deepfake Detector Evaluation (Xception)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_detector.py --detector_path ../training/config/detector/xception.yaml \\
                              --weights_path ../training/weights/xception_best.pth \\
                              --test_dataset FFHQ-FaceFusion-10k_full

  # Quick test with 50 samples
  python evaluate_detector.py --detector_path ../training/config/detector/xception.yaml \\
                              --weights_path ../training/weights/xception_best.pth \\
                              --test_dataset FFHQ-FaceFusion-10k_full \\
                              --max_samples 50
        """
    )
    parser.add_argument(
        '--detector_path', type=str, required=True,
        help='Path to detector YAML configuration file'
    )
    parser.add_argument(
        '--test_dataset', type=str, default='FFHQ-FaceFusion-10k_full',
        help='Name of test dataset (default: FFHQ-FaceFusion-10k_full)'
    )
    parser.add_argument(
        '--weights_path', type=str, required=True,
        help='Path to model weights file (.pth)'
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Limit number of samples for quick testing (e.g. 50)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from last checkpoint'
    )
    return parser.parse_args()


# ==============================================================================
#  EVALUATION
# ==============================================================================

def evaluate_model(model, data_loader, max_samples=None):
    """
    Evaluate model on dataset.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader with test data
        max_samples: Optional limit on number of samples

    Returns:
        tuple: (probabilities, labels) as numpy arrays
    """
    model.eval()
    prediction_lists = []
    label_lists = []
    sample_count = 0

    with torch.no_grad():
        for data_dict in tqdm(data_loader, desc="Evaluation"):
            data, label = data_dict['image'], data_dict['label']
            label = torch.where(label != 0, 1, 0)

            data, label = data.to(DEVICE), label.to(DEVICE)
            data_dict['image'], data_dict['label'] = data, label

            if data_dict.get('mask') is not None:
                data_dict['mask'] = data_dict['mask'].to(DEVICE)
            if data_dict.get('landmark') is not None:
                data_dict['landmark'] = data_dict['landmark'].to(DEVICE)

            predictions = model(data_dict, inference=True)

            batch_labels = label.cpu().numpy().tolist()
            batch_probs = predictions['prob'].cpu().numpy().tolist()

            label_lists.extend(batch_labels)
            prediction_lists.extend(batch_probs)

            sample_count += len(batch_labels)
            if max_samples and sample_count >= max_samples:
                prediction_lists = prediction_lists[:max_samples]
                label_lists = label_lists[:max_samples]
                break

    return np.array(prediction_lists), np.array(label_lists)


# ==============================================================================
#  METRICS
# ==============================================================================

def compute_metrics(y_pred, y_true, threshold=0.5):
    """
    Compute all evaluation metrics.

    Args:
        y_pred: Predicted probabilities (0-1)
        y_true: Actual labels (0 or 1)
        threshold: Decision threshold

    Returns:
        dict: Dictionary with all metrics
    """
    y_pred_binary = (y_pred >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred_binary)
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    # Per-class accuracy
    real_mask = y_true == 0
    fake_mask = y_true == 1
    real_acc = accuracy_score(y_true[real_mask], y_pred_binary[real_mask]) if real_mask.sum() > 0 else 0.0
    fake_acc = accuracy_score(y_true[fake_mask], y_pred_binary[fake_mask]) if fake_mask.sum() > 0 else 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'real_accuracy': float(real_acc),
        'fake_accuracy': float(fake_acc),
        'auc_roc': float(auc),
        'average_precision': float(ap),
        'confusion_matrix': {
            'TP': int(tp), 'TN': int(tn),
            'FP': int(fp), 'FN': int(fn)
        },
        'num_samples': len(y_true),
        'num_real': int(real_mask.sum()),
        'num_fake': int(fake_mask.sum())
    }
    return metrics


# ==============================================================================
#  PER-IMAGE CSV
# ==============================================================================

def save_per_image_csv(image_paths, labels, probabilities, output_path):
    """Save per-image results to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image_path', 'image_name', 'true_label', 'true_class',
            'probability_fake', 'prediction_default', 'predicted_class'
        ])
        for path, label, prob in zip(image_paths, labels, probabilities):
            true_class = 'real' if label == 0 else 'fake'
            pred_default = 1 if prob >= 0.5 else 0
            pred_class = 'real' if pred_default == 0 else 'fake'
            writer.writerow([
                path, os.path.basename(str(path)), int(label), true_class,
                f'{prob:.6f}', pred_default, pred_class
            ])


# ==============================================================================
#  VISUALIZATIONS
# ==============================================================================

def plot_roc_curve(y_pred, y_true, output_dir):
    """Generate ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random baseline')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(y_pred, y_true, output_dir):
    """Generate Precision-Recall curve."""
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(rec_curve, prec_curve, color='green', lw=2,
             label=f'PR curve (AP = {ap:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_pred, y_true, output_dir, threshold=0.5):
    """Generate confusion matrix visualization."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Real', 'Fake'], fontsize=12)
    plt.yticks(tick_marks, ['Real', 'Fake'], fontsize=12)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     fontsize=14,
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
#  CONSOLE OUTPUT
# ==============================================================================

def print_results(metrics):
    """Formatted output of results to console."""
    cm = metrics['confusion_matrix']

    print()
    print("=" * 70)
    print("                     EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total samples: {metrics['num_samples']}")
    print(f"  Real images: {metrics['num_real']}")
    print(f"  Fake images: {metrics['num_fake']}")
    print("-" * 70)
    print(f"Overall Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Real Accuracy:        {metrics['real_accuracy']:.4f} ({metrics['real_accuracy']*100:.2f}%)")
    print(f"Fake Accuracy:        {metrics['fake_accuracy']:.4f} ({metrics['fake_accuracy']*100:.2f}%)")
    print(f"Precision:            {metrics['precision']:.4f}")
    print(f"Recall:               {metrics['recall']:.4f}")
    print(f"F1 Score:             {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:              {metrics['auc_roc']:.4f}")
    print(f"Average Precision:    {metrics['average_precision']:.4f}")
    print("-" * 70)
    print("Confusion Matrix:")
    print("              Predicted")
    print(f"              {'Real':>6}  {'Fake':>6}")
    print(f"Actual Real   {cm['TN']:>6}  {cm['FP']:>6}")
    print(f"       Fake   {cm['FN']:>6}  {cm['TP']:>6}")
    print("=" * 70)


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    """Main function."""
    args = parse_arguments()

    # Load detector config
    with open(args.detector_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Load test config
    test_config_path = os.path.join(PROJECT_ROOT, 'training', 'config', 'test_config.yaml')
    with open(test_config_path, 'r', encoding='utf-8') as f:
        config2 = yaml.safe_load(f)

    config.update(config2)
    config['test_dataset'] = args.test_dataset
    config['weights_path'] = args.weights_path

    # Create output directory: vysledky/{dataset}_{timestamp}/
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dataset_base = args.test_dataset.replace('_full', '')
    output_dir = os.path.join(SCRIPT_DIR, 'vysledky', f'{dataset_base}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Header
    print("=" * 70)
    print("  DEEPFAKE DETECTOR EVALUATION (Xception)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model:            {config['model_name']}")
    print(f"  Weights:          {args.weights_path}")
    print(f"  Dataset:          {args.test_dataset}")
    print(f"  Device:           {DEVICE}")
    print(f"  Max samples:      {args.max_samples or 'all'}")
    print(f"  Output:           {output_dir}")
    print("=" * 70)

    # Prepare dataset
    test_set = DeepfakeAbstractBaseDataset(config=config, mode='test')

    # Extract image paths (before potential subsetting)
    image_paths_all = []
    for frames in test_set.data_dict['image']:
        if isinstance(frames, list):
            image_paths_all.append(frames[0])
        else:
            image_paths_all.append(frames)

    # Apply max_samples limit
    actual_samples = len(test_set)
    if args.max_samples and args.max_samples < len(test_set):
        actual_samples = args.max_samples

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config['test_batchSize'],
        shuffle=False,
        num_workers=0,
        collate_fn=test_set.collate_fn,
        drop_last=False
    )

    print(f"\nDataset loaded: {len(test_set)} samples")
    if args.max_samples:
        print(f"Quick test mode: using {actual_samples} samples")
    print(f"Batch size: {config['test_batchSize']}")

    # Load model
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(DEVICE)

    ckpt = torch.load(args.weights_path, map_location=DEVICE)
    model.load_state_dict(ckpt, strict=False)
    print(f"\n[OK] Model loaded successfully")

    # Evaluate
    print(f"\nStarting evaluation...")
    probabilities, labels = evaluate_model(model, test_loader, args.max_samples)

    # Match image paths to evaluated samples
    image_paths = image_paths_all[:len(probabilities)]

    # Compute metrics
    metrics = compute_metrics(probabilities, labels)

    # Save per-image CSV
    csv_path = os.path.join(output_dir, 'per_image_results.csv')
    save_per_image_csv(image_paths, labels, probabilities, csv_path)
    print(f"\n[OK] Per-image results saved: per_image_results.csv")

    # Save metrics JSON
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save results.pkl
    results_data = {
        'labels': labels,
        'predictions': (probabilities >= 0.5).astype(int),
        'probabilities': probabilities,
        'image_paths': image_paths
    }
    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results_data, f)

    # Save checkpoint_info.pkl
    checkpoint_info = {
        'checkpoint_name': os.path.basename(args.weights_path),
        'timestamp': datetime.now().isoformat(),
        'detector_path': args.detector_path,
        'test_dataset': args.test_dataset,
        'max_samples': args.max_samples,
        'device': str(DEVICE)
    }
    with open(os.path.join(output_dir, 'checkpoint_info.pkl'), 'wb') as f:
        pickle.dump(checkpoint_info, f)

    # Generate plots
    plot_roc_curve(probabilities, labels, output_dir)
    plot_precision_recall_curve(probabilities, labels, output_dir)
    plot_confusion_matrix(probabilities, labels, output_dir)

    # Print results
    print_results(metrics)

    print(f"\n[i] Results saved to: {output_dir}/")
    print(f"   - metrics.json")
    print(f"   - per_image_results.csv")
    print(f"   - results.pkl")
    print(f"   - checkpoint_info.pkl")
    print(f"   - roc_curve.png")
    print(f"   - precision_recall_curve.png")
    print(f"   - confusion_matrix.png")
    print(f"\n[OK] Evaluation completed successfully!")


if __name__ == '__main__':
    main()
