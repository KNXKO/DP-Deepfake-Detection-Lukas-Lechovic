# -*- coding: utf-8 -*-
"""
================================================================================
THRESHOLD ANALYSIS SCRIPT
================================================================================
Author: Diploma Thesis
Description: Recalculate metrics at different thresholds from per_image_results.csv
             without re-running the model.

Usage:
  # Auto compare thresholds 0.3 - 0.9
  python analyze_results.py --results_dir vysledky/FFHQ-FaceFusion-10k_2026-03-11_15-30-45

  # Single threshold
  python analyze_results.py --results_dir <dir> --threshold 0.6

  # Multiple thresholds
  python analyze_results.py --results_dir <dir> --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9
================================================================================
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, average_precision_score,
    confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Threshold Analysis for Deepfake Detection Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: compare thresholds 0.3 to 0.9
  python analyze_results.py --results_dir vysledky/FFHQ-FaceFusion-10k_2026-03-11_15-30-45

  # Single threshold
  python analyze_results.py --results_dir <dir> --threshold 0.6

  # Custom thresholds
  python analyze_results.py --results_dir <dir> --thresholds 0.3 0.5 0.7 0.9
        """
    )
    parser.add_argument(
        '--results_dir', type=str, required=True,
        help='Path to results directory containing per_image_results.csv'
    )
    parser.add_argument(
        '--threshold', type=float, default=None,
        help='Single threshold to evaluate'
    )
    parser.add_argument(
        '--thresholds', type=float, nargs='+', default=None,
        help='List of thresholds to compare'
    )
    return parser.parse_args()


def load_csv(csv_path):
    """Load per_image_results.csv and return labels and probabilities."""
    labels = []
    probabilities = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row['true_label']))
            probabilities.append(float(row['probability_fake']))

    return np.array(labels), np.array(probabilities)


def compute_metrics_at_threshold(y_true, y_pred_prob, threshold):
    """Compute all metrics at a given threshold."""
    y_pred = (y_pred_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)

    real_mask = y_true == 0
    fake_mask = y_true == 1
    real_acc = accuracy_score(y_true[real_mask], y_pred[real_mask]) if real_mask.sum() > 0 else 0.0
    fake_acc = accuracy_score(y_true[fake_mask], y_pred[fake_mask]) if fake_mask.sum() > 0 else 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'threshold': float(threshold),
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


def print_single_threshold(metrics):
    """Print metrics for a single threshold."""
    cm = metrics['confusion_matrix']

    print()
    print("=" * 70)
    print(f"  METRICS AT THRESHOLD {metrics['threshold']:.2f}")
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


def print_comparison_table(all_metrics):
    """Print comparison table for multiple thresholds."""
    print()
    print("=" * 90)
    print("  THRESHOLD COMPARISON TABLE")
    print("=" * 90)
    print(f"  {'Threshold':>9} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} |"
          f" {'F1':>8} | {'Real Acc':>8} | {'Fake Acc':>8}")
    print(f"  {'-' * 82}")
    for m in all_metrics:
        print(f"  {m['threshold']:>9.2f} | {m['accuracy']:>8.4f} | {m['precision']:>9.4f} |"
              f" {m['recall']:>6.4f} | {m['f1_score']:>8.4f} |"
              f" {m['real_accuracy']:>8.4f} | {m['fake_accuracy']:>8.4f}")
    print("=" * 90)


def save_comparison_csv(all_metrics, output_path):
    """Save threshold comparison to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'threshold', 'accuracy', 'precision', 'recall', 'f1_score',
            'real_accuracy', 'fake_accuracy', 'auc_roc', 'average_precision',
            'TP', 'TN', 'FP', 'FN'
        ])
        for m in all_metrics:
            cm = m['confusion_matrix']
            writer.writerow([
                f"{m['threshold']:.2f}",
                f"{m['accuracy']:.4f}", f"{m['precision']:.4f}",
                f"{m['recall']:.4f}", f"{m['f1_score']:.4f}",
                f"{m['real_accuracy']:.4f}", f"{m['fake_accuracy']:.4f}",
                f"{m['auc_roc']:.4f}", f"{m['average_precision']:.4f}",
                cm['TP'], cm['TN'], cm['FP'], cm['FN']
            ])


def plot_threshold_comparison(all_metrics, output_path):
    """Generate threshold comparison chart."""
    thresholds = [m['threshold'] for m in all_metrics]
    accuracies = [m['accuracy'] for m in all_metrics]
    precisions = [m['precision'] for m in all_metrics]
    recalls = [m['recall'] for m in all_metrics]
    f1s = [m['f1_score'] for m in all_metrics]
    real_accs = [m['real_accuracy'] for m in all_metrics]
    fake_accs = [m['fake_accuracy'] for m in all_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, 'b-o', lw=2, label='Accuracy')
    plt.plot(thresholds, precisions, 'g-s', lw=2, label='Precision')
    plt.plot(thresholds, recalls, 'r-^', lw=2, label='Recall')
    plt.plot(thresholds, f1s, 'm-D', lw=2, label='F1 Score')
    plt.plot(thresholds, real_accs, 'c--', lw=2, label='Real Accuracy')
    plt.plot(thresholds, fake_accs, 'y--', lw=2, label='Fake Accuracy')

    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metrics vs. Threshold', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function."""
    args = parse_arguments()

    # Find CSV
    csv_path = os.path.join(args.results_dir, 'per_image_results.csv')
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    print(f"\nLoading results from: {csv_path}")
    y_true, y_pred_prob = load_csv(csv_path)
    print(f"Loaded {len(y_true)} samples ({(y_true == 0).sum()} real, {(y_true == 1).sum()} fake)")

    output_dir = args.results_dir

    if args.threshold is not None:
        # Single threshold mode
        metrics = compute_metrics_at_threshold(y_true, y_pred_prob, args.threshold)
        print_single_threshold(metrics)

        # Save metrics for this threshold
        fname = f"metrics_threshold_{args.threshold:.2f}.json"
        with open(os.path.join(output_dir, fname), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Saved: {fname}")

    else:
        # Multi-threshold comparison
        if args.thresholds:
            thresholds = sorted(args.thresholds)
        else:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        all_metrics = []
        for t in thresholds:
            m = compute_metrics_at_threshold(y_true, y_pred_prob, t)
            all_metrics.append(m)

        # Print table
        print_comparison_table(all_metrics)

        # Save JSON
        json_path = os.path.join(output_dir, 'threshold_comparison.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)

        # Save CSV
        csv_out = os.path.join(output_dir, 'threshold_comparison.csv')
        save_comparison_csv(all_metrics, csv_out)

        # Save plot
        png_path = os.path.join(output_dir, 'threshold_comparison.png')
        plot_threshold_comparison(all_metrics, png_path)

        print(f"\n[OK] Saved:")
        print(f"   - threshold_comparison.json")
        print(f"   - threshold_comparison.csv")
        print(f"   - threshold_comparison.png")

    print(f"\n[OK] Analysis completed!")


if __name__ == '__main__':
    main()
