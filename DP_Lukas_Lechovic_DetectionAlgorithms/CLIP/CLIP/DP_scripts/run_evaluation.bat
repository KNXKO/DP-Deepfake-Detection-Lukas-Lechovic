@echo off
REM ============================================================================
REM  CLIP-BASED UNIVERSAL FAKE IMAGE DETECTOR - EVALUATION
REM ============================================================================
REM  Runs CLIP-based detector evaluation on custom dataset.
REM  Edit the CONFIGURATION section below before first use.
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo             CLIP-BASED UNIVERSAL FAKE IMAGE DETECTOR
echo                    Evaluation Script v2.0
echo ============================================================================
echo.

REM ============================================================================
REM  CONFIGURATION - EDIT THESE PATHS
REM ============================================================================

REM Dataset path (auto-detects real/fake or 0_real/1_fake subfolders)
set DATASET_PATH=C:\Users\diabo\Desktop\FFHQ-FaceFusion-10k

REM OR specify real/fake paths separately (leave DATASET_PATH empty to use these)
set REAL_PATH=
set FAKE_PATH=

REM Model checkpoint
set CHECKPOINT=..\pretrained_weights\fc_weights.pth

REM Output directory (results saved to vysledky/{dataset_name}_{timestamp}/)
set OUTPUT_DIR=vysledky

REM Batch size (lower if GPU memory is limited)
set BATCH_SIZE=32

REM Max samples per class (0 = use all images, e.g. 50 for quick test)
set MAX_SAMPLES=0

REM Dataset name (leave empty for auto-detection from path)
set DATASET_NAME=

REM Conda environment name
set CONDA_ENV=clip

REM ============================================================================
REM  VALIDATION
REM ============================================================================

echo Configuration:

if not "%DATASET_PATH%"=="" (
    echo   Dataset:      %DATASET_PATH%
    if not exist "%DATASET_PATH%" (
        echo [ERROR] Dataset directory not found: %DATASET_PATH%
        pause
        exit /b 1
    )
) else (
    echo   Real images:  %REAL_PATH%
    echo   Fake images:  %FAKE_PATH%
    if "%REAL_PATH%"=="" (
        echo [ERROR] Set DATASET_PATH or both REAL_PATH and FAKE_PATH
        pause
        exit /b 1
    )
    if "%FAKE_PATH%"=="" (
        echo [ERROR] Set DATASET_PATH or both REAL_PATH and FAKE_PATH
        pause
        exit /b 1
    )
    if not exist "%REAL_PATH%" (
        echo [ERROR] Real images directory not found: %REAL_PATH%
        pause
        exit /b 1
    )
    if not exist "%FAKE_PATH%" (
        echo [ERROR] Fake images directory not found: %FAKE_PATH%
        pause
        exit /b 1
    )
)

echo   Checkpoint:   %CHECKPOINT%
echo   Output:       %OUTPUT_DIR%
echo   Batch size:   %BATCH_SIZE%
if not %MAX_SAMPLES%==0 (
    echo   Max samples:  %MAX_SAMPLES% per class [QUICK TEST MODE]
)
echo.

REM ============================================================================
REM  ENVIRONMENT
REM ============================================================================

echo Activating conda environment '%CONDA_ENV%'...
call conda activate %CONDA_ENV%

if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment '%CONDA_ENV%'
    echo Create environment with: conda create -n %CONDA_ENV% python=3.10
    echo Install requirements: pip install torch torchvision scikit-learn tqdm pillow ftfy regex matplotlib
    pause
    exit /b 1
)

REM Set OpenMP duplicate library workaround
set KMP_DUPLICATE_LIB_OK=TRUE

REM ============================================================================
REM  BUILD AND RUN COMMAND
REM ============================================================================

echo.
echo Starting evaluation...
echo.

set CMD=python evaluate_detector.py

if not "%DATASET_PATH%"=="" (
    set CMD=!CMD! --dataset_path "%DATASET_PATH%"
) else (
    set CMD=!CMD! --real_path "%REAL_PATH%" --fake_path "%FAKE_PATH%"
)

set CMD=!CMD! --checkpoint "%CHECKPOINT%"
set CMD=!CMD! --output_dir "%OUTPUT_DIR%"
set CMD=!CMD! --batch_size %BATCH_SIZE%

if not %MAX_SAMPLES%==0 (
    set CMD=!CMD! --max_samples %MAX_SAMPLES%
)

if not "%DATASET_NAME%"=="" (
    set CMD=!CMD! --dataset_name "%DATASET_NAME%"
)

echo Running: !CMD!
echo.
!CMD!

REM ============================================================================
REM  COMPLETION
REM ============================================================================

echo.
echo ============================================================================
echo                        EVALUATION COMPLETE
echo ============================================================================
echo.
echo Results saved to: %OUTPUT_DIR%\
echo.
echo Output files:
echo   - metrics.json               (evaluation metrics)
echo   - per_image_results.csv      (per-image probabilities)
echo   - results.pkl                (numpy arrays for analysis)
echo   - checkpoint_info.pkl        (checkpoint metadata)
echo   - confusion_matrix.png       (confusion matrix plot)
echo   - roc_curve.png              (ROC curve)
echo   - precision_recall_curve.png (PR curve)
echo.
echo To analyze different thresholds:
echo   python analyze_results.py --results_dir %OUTPUT_DIR%\DATASET_TIMESTAMP
echo.

pause
