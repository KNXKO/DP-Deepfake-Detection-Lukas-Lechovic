# -*- coding: utf-8 -*-
"""
================================================================================
CLIP THRESHOLD ANALYSIS SCRIPT
================================================================================
Author: Diploma Thesis Project
Description: Recomputes detection metrics at different thresholds using
             per-image results saved by evaluate_detector.py.

             This script answers the question: "If we change the threshold
             from 0.5 to e.g. 0.6, what are the new accuracy/precision/recall
             values?" - without needing to re-run the model.

Usage:
  # Single threshold
  python analyze_results.py --results_dir vysledky/Dataset_results --threshold 0.6

  # Compare multiple thresholds
  python analyze_results.py --results_dir vysledky/Dataset_results \
                            --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9
================================================================================
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


# ==============================================================================
#  DATA LOADING
# ==============================================================================

def load_per_image_csv(csv_path):
    """
    Load per-image results from CSV.

    Returns:
        Dictionary with arrays: image_paths, image_names, labels, probabilities
    """
    image_paths = []
    image_names = []
    labels = []
    probabilities = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_paths.append(row['image_path'])
            image_names.append(row['image_name'])
            labels.append(int(row['true_label']))
            probabilities.append(float(row['probability_fake']))

    return {
        'image_paths': image_paths,
        'image_names': image_names,
        'labels': np.array(labels),
        'probabilities': np.array(probabilities)
    }


# ==============================================================================
#  THRESHOLD ANALYSIS
# ==============================================================================

def compute_metrics_at_threshold(labels, probabilities, threshold):
    """
    Compute all metrics using a given threshold.

    Args:
        labels: Ground truth labels (0=real, 1=fake)
        probabilities: Model predicted probabilities for class 'fake'
        threshold: Classification threshold (image is 'fake' if prob >= threshold)

    Returns:
        Dictionary with metrics
    """
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    real_mask = labels == 0
    fake_mask = labels == 1
    real_accuracy = accuracy_score(labels[real_mask], predictions[real_mask]) if real_mask.sum() > 0 else 0
    fake_accuracy = accuracy_score(labels[fake_mask], predictions[fake_mask]) if fake_mask.sum() > 0 else 0

    try:
        auc_roc = roc_auc_score(labels, probabilities)
        avg_precision = average_precision_score(labels, probabilities)
    except ValueError:
        auc_roc = 0.0
        avg_precision = 0.0

    cm = confusion_matrix(labels, predictions)

    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'real_accuracy': float(real_accuracy),
        'fake_accuracy': float(fake_accuracy),
        'auc_roc': float(auc_roc),
        'average_precision': float(avg_precision),
        'confusion_matrix': cm.tolist(),
        'num_samples': len(labels),
        'num_real': int(real_mask.sum()),
        'num_fake': int(fake_mask.sum()),
        'num_predicted_real': int((predictions == 0).sum()),
        'num_predicted_fake': int((predictions == 1).sum())
    }


def print_metrics(metrics):
    """Print metrics for a single threshold."""
    print(f"\n{'=' * 70}")
    print(f"  THRESHOLD = {metrics['threshold']:.2f}")
    print(f"{'=' * 70}")
    print(f"  Total samples:       {metrics['num_samples']}")
    print(f"    Real images:       {metrics['num_real']}")
    print(f"    Fake images:       {metrics['num_fake']}")
    print(f"    Predicted real:    {metrics['num_predicted_real']}")
    print(f"    Predicted fake:    {metrics['num_predicted_fake']}")
    print(f"  {'-' * 66}")
    print(f"  Overall Accuracy:    {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Real Accuracy:       {metrics['real_accuracy']:.4f}  ({metrics['real_accuracy']*100:.2f}%)")
    print(f"  Fake Accuracy:       {metrics['fake_accuracy']:.4f}  ({metrics['fake_accuracy']*100:.2f}%)")
    print(f"  Precision:           {metrics['precision']:.4f}")
    print(f"  Recall:              {metrics['recall']:.4f}")
    print(f"  F1 Score:            {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:             {metrics['auc_roc']:.4f}")
    print(f"  Average Precision:   {metrics['average_precision']:.4f}")
    print(f"  {'-' * 66}")
    cm = metrics['confusion_matrix']
    print(f"  Confusion Matrix:      Predicted")
    print(f"                       Real   Fake")
    print(f"    Actual Real      {cm[0][0]:6d} {cm[0][1]:6d}")
    print(f"           Fake      {cm[1][0]:6d} {cm[1][1]:6d}")
    print(f"{'=' * 70}")


# ==============================================================================
#  MULTI-THRESHOLD COMPARISON
# ==============================================================================

def compare_thresholds(labels, probabilities, thresholds, output_dir):
    """
    Compute and compare metrics across multiple thresholds.

    Args:
        labels: Ground truth labels
        probabilities: Predicted probabilities
        thresholds: List of threshold values
        output_dir: Directory to save comparison results
    """
    all_metrics = []

    for t in thresholds:
        m = compute_metrics_at_threshold(labels, probabilities, t)
        all_metrics.append(m)
        print_metrics(m)

    # Print comparison table
    print(f"\n{'=' * 90}")
    print("  THRESHOLD COMPARISON TABLE")
    print(f"{'=' * 90}")
    header = f"  {'Threshold':>9} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'Real Acc':>8} | {'Fake Acc':>8}"
    print(header)
    print(f"  {'-' * 86}")
    for m in all_metrics:
        row = (f"  {m['threshold']:>9.2f} | {m['accuracy']:>8.4f} | {m['precision']:>9.4f} | "
               f"{m['recall']:>6.4f} | {m['f1_score']:>6.4f} | {m['real_accuracy']:>8.4f} | {m['fake_accuracy']:>8.4f}")
        print(row)
    print(f"{'=' * 90}")

    # Save comparison JSON
    comparison_path = os.path.join(output_dir, 'threshold_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved: {comparison_path}")

    # Save comparison CSV
    csv_path = os.path.join(output_dir, 'threshold_comparison.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['threshold', 'accuracy', 'precision', 'recall', 'f1_score',
                         'real_accuracy', 'fake_accuracy', 'auc_roc', 'average_precision'])
        for m in all_metrics:
            writer.writerow([m['threshold'], m['accuracy'], m['precision'], m['recall'],
                             m['f1_score'], m['real_accuracy'], m['fake_accuracy'],
                             m['auc_roc'], m['average_precision']])
    print(f"Saved: {csv_path}")

    # Plot threshold comparison
    plot_threshold_comparison(all_metrics, output_dir)

    return all_metrics


def plot_threshold_comparison(all_metrics, output_dir):
    """Plot metrics as a function of threshold."""
    thresholds = [m['threshold'] for m in all_metrics]
    accuracies = [m['accuracy'] for m in all_metrics]
    precisions = [m['precision'] for m in all_metrics]
    recalls = [m['recall'] for m in all_metrics]
    f1s = [m['f1_score'] for m in all_metrics]
    real_accs = [m['real_accuracy'] for m in all_metrics]
    fake_accs = [m['fake_accuracy'] for m in all_metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, accuracies, 'b-o', label='Accuracy', linewidth=2)
    ax.plot(thresholds, precisions, 'g-s', label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, 'r-^', label='Recall', linewidth=2)
    ax.plot(thresholds, f1s, 'm-D', label='F1 Score', linewidth=2)
    ax.plot(thresholds, real_accs, 'c--', label='Real Accuracy', linewidth=1.5, alpha=0.7)
    ax.plot(thresholds, fake_accs, 'y--', label='Fake Accuracy', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('CLIP Detector - Metrics vs. Threshold', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(thresholds) - 0.02, max(thresholds) + 0.02])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'threshold_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CLIP Threshold Analysis - Recompute metrics at different thresholds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recompute at threshold 0.6
  python analyze_results.py --results_dir vysledky/Dataset_results --threshold 0.6

  # Compare multiple thresholds
  python analyze_results.py --results_dir vysledky/Dataset_results \\
                            --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9
        """
    )
    parser.add_argument('--results_dir', required=True, type=str,
                        help='Directory containing per_image_results.csv')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Single threshold to evaluate')
    parser.add_argument('--thresholds', type=float, nargs='+', default=None,
                        help='Multiple thresholds to compare')

    args = parser.parse_args()

    # Find CSV file
    csv_path = os.path.join(args.results_dir, 'per_image_results.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: Per-image results not found: {csv_path}")
        print("Run evaluate_detector.py first to generate per-image results.")
        sys.exit(1)

    # Load data
    print(f"Loading per-image results from: {csv_path}")
    data = load_per_image_csv(csv_path)
    print(f"Loaded {len(data['labels'])} images "
          f"({(data['labels'] == 0).sum()} real, {(data['labels'] == 1).sum()} fake)")

    # Determine thresholds
    if args.thresholds:
        thresholds = sorted(args.thresholds)
        compare_thresholds(data['labels'], data['probabilities'], thresholds, args.results_dir)
    elif args.threshold is not None:
        metrics = compute_metrics_at_threshold(data['labels'], data['probabilities'], args.threshold)
        print_metrics(metrics)

        # Save single-threshold metrics
        out_path = os.path.join(args.results_dir, f'metrics_threshold_{args.threshold:.2f}.json')
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved: {out_path}")
    else:
        # Default: compare common thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        compare_thresholds(data['labels'], data['probabilities'], thresholds, args.results_dir)


if __name__ == '__main__':
    main()
