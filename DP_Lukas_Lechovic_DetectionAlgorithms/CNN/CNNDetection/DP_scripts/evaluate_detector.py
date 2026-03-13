"""
CNNDetection - Evaluation Script for Deepfake Detection
========================================================

This script evaluates a pre-trained CNNDetection model on a custom dataset
containing real and fake (AI-generated) images. It computes comprehensive
metrics including accuracy, precision, recall, F1-score, specificity,
AUC-ROC, and average precision.

Based on: "CNN-generated images are surprisingly easy to spot...for now"
          Wang et al., CVPR 2020
          https://github.com/PeterWang512/CNNDetection

Usage:
    python evaluate_detector.py -d <dataset_path> -m <model_path>

Example:
    python evaluate_detector.py -d C:/MyDataset -m ../weights/blur_jpg_prob0.5.pth

Dataset structure:
    dataset/
        real/       (or 0_real/)
        fake/       (or 1_fake/)

Author: Diploma Thesis Project
"""

import os
import sys
import argparse
import csv
import json
import pickle
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from networks.resnet import resnet50


# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path):
    """
    Load pre-trained CNNDetection model from checkpoint.

    Args:
        model_path: Path to .pth checkpoint file

    Returns:
        Loaded model in evaluation mode
    """
    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')

    if 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    return model


def get_images_from_folder(folder_path):
    """
    Get all image files from a directory.

    Args:
        folder_path: Path to folder containing images

    Returns:
        List of image file paths
    """
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    images = []

    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            if any(f.lower().endswith(ext) for ext in extensions):
                images.append(os.path.join(folder_path, f))

    return sorted(images)


def predict_image(model, image_path, use_cuda=True):
    """
    Get model prediction for a single image.

    Args:
        model: Loaded CNNDetection model
        image_path: Path to image file
        use_cuda: Whether to use GPU

    Returns:
        Probability score (0=real, 1=fake)
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        if use_cuda and torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()

        return prob
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def find_dataset_folders(dataroot):
    """
    Find real and fake image folders in dataset.
    Supports multiple naming conventions.

    Args:
        dataroot: Root path of dataset

    Returns:
        Tuple of (real_path, fake_path)
    """
    real_folders = ['0_real', 'real', '0_Real', 'Real', 'REAL']
    fake_folders = ['1_fake', 'fake', '1_Fake', 'Fake', 'FAKE']

    real_path = None
    fake_path = None

    for rf in real_folders:
        path = os.path.join(dataroot, rf)
        if os.path.exists(path):
            real_path = path
            break

    for ff in fake_folders:
        path = os.path.join(dataroot, ff)
        if os.path.exists(path):
            fake_path = path
            break

    return real_path, fake_path


def compute_metrics(y_true, y_pred, y_pred_prob):
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Binary predictions
        y_pred_prob: Probability predictions

    Returns:
        Dictionary with all metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    try:
        auc_roc = roc_auc_score(y_true, y_pred_prob)
    except:
        auc_roc = 0.0

    try:
        ap = average_precision_score(y_true, y_pred_prob)
    except:
        ap = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'auc_roc': auc_roc,
        'average_precision': ap,
        'true_negative': int(tn),
        'true_positive': int(tp),
        'false_positive': int(fp),
        'false_negative': int(fn)
    }


def print_results(metrics, num_real, num_fake):
    """Print formatted evaluation results."""
    total = num_real + num_fake

    print()
    print("=" * 70)
    print("                     EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total samples: {total}")
    print(f"  Real images: {num_real}")
    print(f"  Fake images: {num_fake}")
    print("-" * 70)
    print(f"Overall Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Real Accuracy:        {metrics['specificity']:.4f} ({metrics['specificity'] * 100:.2f}%)")
    print(f"Fake Accuracy:        {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"Precision:            {metrics['precision']:.4f}")
    print(f"Recall:               {metrics['recall']:.4f}")
    print(f"F1 Score:             {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:              {metrics['auc_roc']:.4f}")
    print(f"Average Precision:    {metrics['average_precision']:.4f}")
    print("-" * 70)
    print("Confusion Matrix:")
    print("              Predicted")
    print("              Real  Fake")
    print(f"Actual Real  {metrics['true_negative']:>5}  {metrics['false_positive']:>4}")
    print(f"       Fake  {metrics['false_negative']:>5}  {metrics['true_positive']:>4}")
    print("=" * 70)


def save_metrics_json(metrics, args, num_real, num_fake, output_path):
    """Save metrics to JSON file."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'dataset': args.dataroot,
        'threshold': args.threshold,
        'dataset_info': {
            'num_real': num_real,
            'num_fake': num_fake,
            'total': num_real + num_fake
        },
        'metrics': {
            'accuracy': round(metrics['accuracy'] * 100, 2),
            'precision': round(metrics['precision'] * 100, 2),
            'recall': round(metrics['recall'] * 100, 2),
            'f1_score': round(metrics['f1_score'] * 100, 2),
            'real_accuracy': round(metrics['specificity'] * 100, 2),
            'fake_accuracy': round(metrics['recall'] * 100, 2),
            'specificity': round(metrics['specificity'] * 100, 2),
            'auc_roc': round(metrics['auc_roc'] * 100, 2),
            'average_precision': round(metrics['average_precision'] * 100, 2)
        },
        'confusion_matrix': {
            'true_negative': metrics['true_negative'],
            'true_positive': metrics['true_positive'],
            'false_positive': metrics['false_positive'],
            'false_negative': metrics['false_negative']
        },
        'num_samples': num_real + num_fake,
        'num_real': num_real,
        'num_fake': num_fake
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def save_per_image_csv(image_paths, y_true, y_pred_prob, threshold, output_path):
    """Save per-image results to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image_path', 'image_name', 'true_label', 'true_class',
            'probability_fake', 'prediction_default', 'predicted_class'
        ])
        for i in range(len(image_paths)):
            img_path = image_paths[i]
            img_name = os.path.basename(img_path)
            true_label = int(y_true[i])
            true_class = 'fake' if true_label == 1 else 'real'
            prob = float(y_pred_prob[i])
            pred = 1 if prob > threshold else 0
            pred_class = 'fake' if pred == 1 else 'real'
            writer.writerow([
                img_path, img_name, true_label, true_class,
                f"{prob:.6f}", pred, pred_class
            ])


def save_results_pkl(y_true, y_pred, y_pred_prob, image_paths, output_path):
    """Save raw results to pickle."""
    data = {
        'labels': y_true,
        'predictions': y_pred,
        'probabilities': y_pred_prob,
        'image_paths': image_paths
    }
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def save_checkpoint_info(model_path, output_path):
    """Save checkpoint info to pickle."""
    info = {
        'checkpoint': model_path,
        'timestamp': datetime.now().isoformat()
    }
    with open(output_path, 'wb') as f:
        pickle.dump(info, f)


def plot_confusion_matrix(metrics, output_path):
    """Plot and save confusion matrix."""
    cm = np.array([
        [metrics['true_negative'], metrics['false_positive']],
        [metrics['false_negative'], metrics['true_positive']]
    ])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ['Real', 'Fake']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label', xlabel='Predicted Label',
           title='Confusion Matrix')

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center', fontsize=14,
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_roc_curve(y_true, y_pred_prob, auc_value, output_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {auc_value:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_precision_recall_curve(y_true, y_pred_prob, ap_value, output_path):
    """Plot and save Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_true, y_pred_prob)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(rec, prec, color='green', lw=2,
            label=f'PR curve (AP = {ap_value:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate CNNDetection model on custom dataset'
    )
    parser.add_argument('-d', '--dataroot', required=True,
                        help='Path to dataset (containing real/ and fake/ folders)')
    parser.add_argument('-m', '--model', required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max images per class for quick testing')
    parser.add_argument('--output', type=str, default=None,
                        help='Custom output directory path')
    args = parser.parse_args()

    # Find dataset folders
    real_path, fake_path = find_dataset_folders(args.dataroot)

    if real_path is None or fake_path is None:
        print(f"ERROR: Could not find real/fake folders in {args.dataroot}")
        print(f"Available folders: {os.listdir(args.dataroot)}")
        sys.exit(1)

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    print(f"Real images: {real_path}")
    print(f"Fake images: {fake_path}")

    real_images = get_images_from_folder(real_path)
    fake_images = get_images_from_folder(fake_path)

    # Quick test mode
    if args.max_samples is not None:
        real_images = real_images[:args.max_samples]
        fake_images = fake_images[:args.max_samples]
        print(f"[QUICK TEST MODE] Limited to {args.max_samples} samples per class")

    print(f"Number of REAL images: {len(real_images)}")
    print(f"Number of FAKE images: {len(fake_images)}")
    print(f"Total: {len(real_images) + len(fake_images)}")
    print()

    # Load model
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    print(f"Model: {args.model}")
    model = load_model(args.model)
    print("Model loaded successfully!")
    print()

    # Create output directory
    dataset_name = os.path.basename(args.dataroot.rstrip('/\\'))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(
            os.path.dirname(__file__), 'vysledky',
            f"{dataset_name}_{timestamp}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # Run predictions
    print("=" * 70)
    print("RUNNING PREDICTIONS")
    print("=" * 70)

    use_cuda = torch.cuda.is_available()
    y_true = []
    y_pred_prob = []
    image_paths = []

    # Process real images (label = 0)
    print("Processing REAL images...")
    for img_path in tqdm(real_images, desc="Real"):
        prob = predict_image(model, img_path, use_cuda)
        if prob is not None:
            y_true.append(0)
            y_pred_prob.append(prob)
            image_paths.append(img_path)

    # Process fake images (label = 1)
    print("Processing FAKE images...")
    for img_path in tqdm(fake_images, desc="Fake"):
        prob = predict_image(model, img_path, use_cuda)
        if prob is not None:
            y_true.append(1)
            y_pred_prob.append(prob)
            image_paths.append(img_path)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = (y_pred_prob > args.threshold).astype(int)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_pred_prob)

    num_real = len(real_images)
    num_fake = len(fake_images)

    # Print results
    print_results(metrics, num_real, num_fake)

    # Save all outputs
    print()
    print("Saving results...")

    # 1. metrics.json
    metrics_path = os.path.join(output_dir, 'metrics.json')
    save_metrics_json(metrics, args, num_real, num_fake, metrics_path)
    print(f"  metrics.json -> {metrics_path}")

    # 2. per_image_results.csv
    csv_path = os.path.join(output_dir, 'per_image_results.csv')
    save_per_image_csv(image_paths, y_true, y_pred_prob, args.threshold, csv_path)
    print(f"  per_image_results.csv -> {csv_path}")

    # 3. results.pkl
    pkl_path = os.path.join(output_dir, 'results.pkl')
    save_results_pkl(y_true, y_pred, y_pred_prob, image_paths, pkl_path)
    print(f"  results.pkl -> {pkl_path}")

    # 4. checkpoint_info.pkl
    ckpt_path = os.path.join(output_dir, 'checkpoint_info.pkl')
    save_checkpoint_info(args.model, ckpt_path)
    print(f"  checkpoint_info.pkl -> {ckpt_path}")

    # 5. confusion_matrix.png
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(metrics, cm_path)
    print(f"  confusion_matrix.png -> {cm_path}")

    # 6. roc_curve.png
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(y_true, y_pred_prob, metrics['auc_roc'], roc_path)
    print(f"  roc_curve.png -> {roc_path}")

    # 7. precision_recall_curve.png
    pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plot_precision_recall_curve(y_true, y_pred_prob, metrics['average_precision'], pr_path)
    print(f"  precision_recall_curve.png -> {pr_path}")

    print()
    print(f"All results saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
