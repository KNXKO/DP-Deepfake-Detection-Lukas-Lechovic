# CLIP-Based Universal Fake Image Detector

Evaluation scripts for the CLIP-based Universal Fake Image Detector.

## Overview

This directory contains scripts for evaluating the Universal Fake Image Detector
on custom datasets. The detector is based on CLIP (Contrastive Language-Image
Pre-Training) features and was proposed by Ojha et al. at CVPR 2023.

## Files

| File | Description |
|------|-------------|
| `evaluate_clip_detector.py` | Main evaluation script with metrics computation |
| `run_clip_evaluation.bat` | Full evaluation on entire dataset |
| `run_quick_test.bat` | Quick test with 100 samples per class |

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- Conda environment `clip`

### Dependencies

```
torch
torchvision
scikit-learn
numpy
pillow
tqdm
ftfy
regex
```

## Dataset Structure

```
MyDataset/
    real/
        image001.png
        image002.jpg
        ...
    fake/
        fake001.png
        fake002.jpg
        ...
```

## Usage

### Quick Test (100 samples)

```batch
run_quick_test.bat
```

### Full Evaluation

```batch
run_clip_evaluation.bat
```

### Custom Evaluation

```batch
python evaluate_clip_detector.py ^
    --real_path C:\path\to\real ^
    --fake_path C:\path\to\fake ^
    --output_dir results ^
    --batch_size 32
```

## Output Metrics

The evaluation produces the following metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: True positive rate (sensitivity)
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **AUC-ROC**: Area under ROC curve
- **Average Precision**: Area under precision-recall curve

## Reference

```bibtex
@inproceedings{ojha2023fakedetect,
    title={Towards Universal Fake Image Detectors that Generalize Across
           Generative Models},
    author={Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
    booktitle={CVPR},
    year={2023}
}
```

## License

MIT License - See UniversalFakeDetect repository for details.
