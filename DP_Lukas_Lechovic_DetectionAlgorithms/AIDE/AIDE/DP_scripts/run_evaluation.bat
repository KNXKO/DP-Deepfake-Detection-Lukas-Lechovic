@echo off
REM ============================================================================
REM  AIDE DETECTOR EVALUATION
REM ============================================================================
REM  Author: Diploma Thesis Project
REM  Description: Runs AIDE detector evaluation on prepared dataset
REM
REM  Requirements:
REM    - Conda environment 'aide' with all dependencies installed
REM    - Prepared dataset (run prepare_dataset.py first)
REM    - Model checkpoints in ../AIDE/checkpoints/
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo                        AIDE DETECTOR EVALUATION
echo                 AI-generated Image Detection Evaluation
echo ============================================================================
echo.

REM ============================================================================
REM  CONFIGURATION - Modify these paths as needed
REM ============================================================================

set DATASET_PATH=C:\Users\diabo\Desktop\FFHQ-FaceFusion-10k
set CHECKPOINT=C:\Users\diabo\Desktop\AIDE\AIDE\checkpoints\GenImage_train.pth
set RESNET_PATH=C:\Users\diabo\Desktop\AIDE\AIDE\checkpoints\resnet50.pth
set CONVNEXT_PATH=C:\Users\diabo\Desktop\AIDE\AIDE\checkpoints\open_clip_pytorch_model.bin
set OUTPUT_DIR=vysledky
set BATCH_SIZE=8
set NUM_WORKERS=4

REM ============================================================================
REM  VALIDATION
REM ============================================================================

echo Configuration:
echo   Dataset:     %DATASET_PATH%
echo   Checkpoint:  %CHECKPOINT%
echo   Output:      %OUTPUT_DIR%
echo   Batch Size:  %BATCH_SIZE%
echo.
echo ----------------------------------------------------------------------------

REM Check if dataset exists
if not exist "%DATASET_PATH%" (
    echo [ERROR] Dataset not found: %DATASET_PATH%
    echo         Run prepare_dataset.py first to prepare your dataset.
    pause
    exit /b 1
)

REM Check if checkpoint exists
if not exist "%CHECKPOINT%" (
    echo [ERROR] Checkpoint not found: %CHECKPOINT%
    echo         Download the checkpoint from AIDE Model Zoo.
    pause
    exit /b 1
)

REM ============================================================================
REM  ENVIRONMENT ACTIVATION
REM ============================================================================

echo Activating conda environment 'aide'...
call conda activate aide

if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment 'aide'
    echo         Create it with: conda create -n aide python=3.10 -y
    pause
    exit /b 1
)

REM ============================================================================
REM  EVALUATION
REM ============================================================================

echo.
echo Starting evaluation...
echo This may take several hours depending on dataset size.
echo.

python evaluate_detector.py ^
    --eval_data_path %DATASET_PATH% ^
    --checkpoint %CHECKPOINT% ^
    --resnet_path %RESNET_PATH% ^
    --convnext_path %CONVNEXT_PATH% ^
    --output_dir %OUTPUT_DIR% ^
    --batch_size %BATCH_SIZE% ^
    --num_workers %NUM_WORKERS%

REM ============================================================================
REM  RESULTS
REM ============================================================================

echo.
echo ============================================================================
echo                           EVALUATION COMPLETE
echo ============================================================================
echo.
echo Results saved to: %OUTPUT_DIR%\  (folder named by dataset + timestamp)
echo.
echo Output files:
echo   - metrics.json              (numeric metrics)
echo   - per_image_results.csv     (per-image probabilities for threshold analysis)
echo   - confusion_matrix.png      (confusion matrix visualization)
echo   - roc_curve.png             (ROC curve with AUC)
echo   - precision_recall_curve.png (Precision-Recall curve)
echo.
echo To recompute metrics at a different threshold (e.g. 0.6):
echo   python analyze_results.py --results_dir %OUTPUT_DIR%\FOLDER_NAME --threshold 0.6
echo.
echo To compare multiple thresholds:
echo   python analyze_results.py --results_dir %OUTPUT_DIR%\FOLDER_NAME --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9
echo.
echo ============================================================================

pause
