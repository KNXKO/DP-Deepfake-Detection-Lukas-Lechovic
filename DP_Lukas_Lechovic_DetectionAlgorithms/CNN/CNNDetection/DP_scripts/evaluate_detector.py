import os
import sys
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
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
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # AUC and AP
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

    print()
    print("=" * 70)
    print("RESULTS - MAIN METRICS")
    print("=" * 70)
    print(f"Accuracy:                       {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:                      {metrics['precision'] * 100:.2f}%")
    print(f"Recall (Sensitivity):           {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score:                       {metrics['f1_score'] * 100:.2f}%")
    print(f"Specificity:                    {metrics['specificity'] * 100:.2f}%")
    print(f"AUC-ROC:                        {metrics['auc_roc'] * 100:.2f}%")
    print(f"Average Precision (AP):         {metrics['average_precision'] * 100:.2f}%")
    print("=" * 70)

    print()
    print("ACCURACY BY IMAGE TYPE:")
    print("-" * 70)
    print(f"Accuracy on REAL images (Specificity): {metrics['specificity'] * 100:.2f}%")
    print(f"Accuracy on FAKE images (Recall):      {metrics['recall'] * 100:.2f}%")
    print(f"Overall Accuracy:                      {metrics['accuracy'] * 100:.2f}%")

    print()
    print("=" * 70)
    print("DETAILED STATISTICS")
    print("=" * 70)
    print(f"Number of real images:                 {num_real}")
    print(f"Number of fake images:                 {num_fake}")
    print(f"Total images:                          {num_real + num_fake}")
    print("-" * 70)
    print(f"True Negative (correct real):          {metrics['true_negative']}")
    print(f"True Positive (correct fake):          {metrics['true_positive']}")
    print(f"False Positive (real marked as fake):  {metrics['false_positive']}")
    print(f"False Negative (fake marked as real):  {metrics['false_negative']}")
    print("=" * 70)

    print()
    print("METRIC DEFINITIONS:")
    print("-" * 70)
    print("Accuracy:    Overall correctness (correct / total)")
    print("Precision:   Of predicted FAKE, how many were actually FAKE")
    print("Recall:      Of all FAKE images, how many were detected")
    print("F1-Score:    Harmonic mean of Precision and Recall")
    print("Specificity: Of all REAL images, how many were correctly classified")
    print("AUC-ROC:     Area under ROC curve (overall quality)")
    print("AP:          Average Precision (detection quality)")
    print("=" * 70)


def save_results(metrics, args, num_real, num_fake, output_path):
    """Save results to JSON file."""

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
            'specificity': round(metrics['specificity'] * 100, 2),
            'auc_roc': round(metrics['auc_roc'] * 100, 2),
            'average_precision': round(metrics['average_precision'] * 100, 2)
        },
        'confusion_matrix': {
            'true_negative': metrics['true_negative'],
            'true_positive': metrics['true_positive'],
            'false_positive': metrics['false_positive'],
            'false_negative': metrics['false_negative']
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


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
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
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

    # Run predictions
    print("=" * 70)
    print("RUNNING PREDICTIONS")
    print("=" * 70)

    use_cuda = torch.cuda.is_available()
    y_true = []
    y_pred_prob = []

    # Process real images (label = 0)
    print("Processing REAL images...")
    for img_path in tqdm(real_images, desc="Real"):
        prob = predict_image(model, img_path, use_cuda)
        if prob is not None:
            y_true.append(0)
            y_pred_prob.append(prob)

    # Process fake images (label = 1)
    print("Processing FAKE images...")
    for img_path in tqdm(fake_images, desc="Fake"):
        prob = predict_image(model, img_path, use_cuda)
        if prob is not None:
            y_true.append(1)
            y_pred_prob.append(prob)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = (y_pred_prob > args.threshold).astype(int)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_pred_prob)

    # Print results
    print_results(metrics, len(real_images), len(fake_images))

    # Save results if output path specified
    if args.output:
        save_results(metrics, args, len(real_images), len(fake_images), args.output)
    else:
        # Auto-generate output path
        dataset_name = os.path.basename(args.dataroot.rstrip('/\\'))
        output_dir = os.path.join(os.path.dirname(__file__), 'vysledky', dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'metrics.json')
        save_results(metrics, args, len(real_images), len(fake_images), output_path)


if __name__ == '__main__':
    main()
