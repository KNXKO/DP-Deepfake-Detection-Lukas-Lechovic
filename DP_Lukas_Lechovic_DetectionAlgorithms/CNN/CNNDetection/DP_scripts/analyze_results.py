"""
CNNDetection - Threshold Analysis Script
=========================================

Loads per_image_results.csv and recomputes metrics at different thresholds
without re-running the model.

Usage:
    python analyze_results.py <results_dir>
    python analyze_results.py <results_dir> --threshold 0.6
    python analyze_results.py <results_dir> --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9

Author: Diploma Thesis Project
"""

import os
import sys
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


def load_per_image_csv(csv_path):
    """Load per-image results from CSV."""
    image_paths = []
    y_true = []
    y_pred_prob = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_paths.append(row['image_path'])
            y_true.append(int(row['true_label']))
            y_pred_prob.append(float(row['probability_fake']))

    return np.array(y_true), np.array(y_pred_prob), image_paths


def compute_metrics_at_threshold(y_true, y_pred_prob, threshold):
    """Compute metrics at a given threshold."""
    y_pred = (y_pred_prob > threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    try:
        auc_roc = roc_auc_score(y_true, y_pred_prob)
    except:
        auc_roc = 0.0

    try:
        ap = average_precision_score(y_true, y_pred_prob)
    except:
        ap = 0.0

    num_real = int(np.sum(y_true == 0))
    num_fake = int(np.sum(y_true == 1))

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'real_accuracy': specificity,
        'fake_accuracy': recall,
        'specificity': specificity,
        'auc_roc': auc_roc,
        'average_precision': ap,
        'true_negative': int(tn),
        'true_positive': int(tp),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'num_real': num_real,
        'num_fake': num_fake,
        'num_samples': num_real + num_fake
    }


def print_single_threshold(m):
    """Print metrics for a single threshold."""
    print()
    print("=" * 70)
    print("                     EVALUATION RESULTS")
    print("=" * 70)
    print(f"Threshold: {m['threshold']:.2f}")
    print(f"Total samples: {m['num_samples']}")
    print(f"  Real images: {m['num_real']}")
    print(f"  Fake images: {m['num_fake']}")
    print("-" * 70)
    print(f"Overall Accuracy:     {m['accuracy']:.4f} ({m['accuracy'] * 100:.2f}%)")
    print(f"Real Accuracy:        {m['real_accuracy']:.4f} ({m['real_accuracy'] * 100:.2f}%)")
    print(f"Fake Accuracy:        {m['fake_accuracy']:.4f} ({m['fake_accuracy'] * 100:.2f}%)")
    print(f"Precision:            {m['precision']:.4f}")
    print(f"Recall:               {m['recall']:.4f}")
    print(f"F1 Score:             {m['f1_score']:.4f}")
    print(f"AUC-ROC:              {m['auc_roc']:.4f}")
    print(f"Average Precision:    {m['average_precision']:.4f}")
    print("-" * 70)
    print("Confusion Matrix:")
    print("              Predicted")
    print("              Real  Fake")
    print(f"Actual Real  {m['true_negative']:>5}  {m['false_positive']:>4}")
    print(f"       Fake  {m['false_negative']:>5}  {m['true_positive']:>4}")
    print("=" * 70)


def print_comparison_table(all_metrics):
    """Print threshold comparison table."""
    print()
    print("=" * 90)
    print("  THRESHOLD COMPARISON TABLE")
    print("=" * 90)
    print(f"  {'Threshold':>9} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'Real Acc':>8} | {'Fake Acc':>8}")
    print("  " + "-" * 82)
    for m in all_metrics:
        print(f"  {m['threshold']:>9.2f} | {m['accuracy']:>8.4f} | {m['precision']:>9.4f} | "
              f"{m['recall']:>6.4f} | {m['f1_score']:>6.4f} | {m['real_accuracy']:>8.4f} | {m['fake_accuracy']:>8.4f}")
    print("=" * 90)


def save_comparison_json(all_metrics, output_path):
    """Save threshold comparison to JSON."""
    data = {}
    for m in all_metrics:
        key = f"{m['threshold']:.2f}"
        data[key] = {
            'accuracy': round(m['accuracy'] * 100, 2),
            'precision': round(m['precision'] * 100, 2),
            'recall': round(m['recall'] * 100, 2),
            'f1_score': round(m['f1_score'] * 100, 2),
            'real_accuracy': round(m['real_accuracy'] * 100, 2),
            'fake_accuracy': round(m['fake_accuracy'] * 100, 2),
            'auc_roc': round(m['auc_roc'] * 100, 2),
            'average_precision': round(m['average_precision'] * 100, 2),
            'confusion_matrix': {
                'true_negative': m['true_negative'],
                'true_positive': m['true_positive'],
                'false_positive': m['false_positive'],
                'false_negative': m['false_negative']
            }
        }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def save_comparison_csv(all_metrics, output_path):
    """Save threshold comparison to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'threshold', 'accuracy', 'precision', 'recall', 'f1_score',
            'real_accuracy', 'fake_accuracy', 'auc_roc', 'average_precision',
            'true_negative', 'true_positive', 'false_positive', 'false_negative'
        ])
        for m in all_metrics:
            writer.writerow([
                f"{m['threshold']:.2f}",
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1_score']:.4f}",
                f"{m['real_accuracy']:.4f}",
                f"{m['fake_accuracy']:.4f}",
                f"{m['auc_roc']:.4f}",
                f"{m['average_precision']:.4f}",
                m['true_negative'],
                m['true_positive'],
                m['false_positive'],
                m['false_negative']
            ])


def plot_threshold_comparison(all_metrics, output_path):
    """Plot metrics vs threshold."""
    thresholds = [m['threshold'] for m in all_metrics]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.plot(thresholds, [m['accuracy'] for m in all_metrics],
            'b-o', label='Accuracy')
    ax.plot(thresholds, [m['precision'] for m in all_metrics],
            'g-s', label='Precision')
    ax.plot(thresholds, [m['recall'] for m in all_metrics],
            'r-^', label='Recall')
    ax.plot(thresholds, [m['f1_score'] for m in all_metrics],
            'm-D', label='F1')
    ax.plot(thresholds, [m['real_accuracy'] for m in all_metrics],
            'c--', label='Real Acc')
    ax.plot(thresholds, [m['fake_accuracy'] for m in all_metrics],
            'y--', label='Fake Acc')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs. Threshold')
    ax.set_ylim([0.0, 1.05])
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze CNNDetection results at different thresholds'
    )
    parser.add_argument('results_dir',
                        help='Path to results directory containing per_image_results.csv')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Single threshold to evaluate')
    parser.add_argument('--thresholds', type=float, nargs='+', default=None,
                        help='Multiple thresholds to compare')
    args = parser.parse_args()

    # Find CSV file
    csv_path = os.path.join(args.results_dir, 'per_image_results.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find {csv_path}")
        sys.exit(1)

    print(f"Loading results from: {csv_path}")
    y_true, y_pred_prob, image_paths = load_per_image_csv(csv_path)
    print(f"Loaded {len(y_true)} samples")

    output_dir = args.results_dir

    if args.threshold is not None:
        # Single threshold mode
        m = compute_metrics_at_threshold(y_true, y_pred_prob, args.threshold)
        print_single_threshold(m)

        # Save single threshold metrics
        t_str = f"{args.threshold:.2f}"
        out_path = os.path.join(output_dir, f'metrics_threshold_{t_str}.json')
        single_data = {
            'threshold': args.threshold,
            'accuracy': round(m['accuracy'] * 100, 2),
            'precision': round(m['precision'] * 100, 2),
            'recall': round(m['recall'] * 100, 2),
            'f1_score': round(m['f1_score'] * 100, 2),
            'real_accuracy': round(m['real_accuracy'] * 100, 2),
            'fake_accuracy': round(m['fake_accuracy'] * 100, 2),
            'auc_roc': round(m['auc_roc'] * 100, 2),
            'average_precision': round(m['average_precision'] * 100, 2),
            'confusion_matrix': {
                'true_negative': m['true_negative'],
                'true_positive': m['true_positive'],
                'false_positive': m['false_positive'],
                'false_negative': m['false_negative']
            }
        }
        with open(out_path, 'w') as f:
            json.dump(single_data, f, indent=2)
        print(f"\nSaved: {out_path}")

    else:
        # Multiple thresholds mode
        if args.thresholds is not None:
            thresholds = sorted(args.thresholds)
        else:
            # Default: 0.3 to 0.9
            thresholds = [round(t, 1) for t in np.arange(0.3, 1.0, 0.1)]

        all_metrics = []
        for t in thresholds:
            m = compute_metrics_at_threshold(y_true, y_pred_prob, t)
            all_metrics.append(m)

        print_comparison_table(all_metrics)

        # Save outputs
        json_path = os.path.join(output_dir, 'threshold_comparison.json')
        save_comparison_json(all_metrics, json_path)
        print(f"\nSaved: {json_path}")

        csv_out_path = os.path.join(output_dir, 'threshold_comparison.csv')
        save_comparison_csv(all_metrics, csv_out_path)
        print(f"Saved: {csv_out_path}")

        png_path = os.path.join(output_dir, 'threshold_comparison.png')
        plot_threshold_comparison(all_metrics, png_path)
        print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
