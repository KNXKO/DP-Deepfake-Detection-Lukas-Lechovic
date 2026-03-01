@echo off
REM ============================================================================
REM  CLIP-BASED UNIVERSAL FAKE IMAGE DETECTOR - EVALUATION
REM ============================================================================
REM
REM  Requirements:
REM    - Conda environment 'clip' with PyTorch and dependencies
REM    - UniversalFakeDetect repository with pretrained weights
REM    - Dataset with separate real/fake image directories
REM
REM  Reference:
REM    Ojha et al. "Towards Universal Fake Image Detectors that Generalize
REM    Across Generative Models" (CVPR 2023)
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo             CLIP-BASED UNIVERSAL FAKE IMAGE DETECTOR
echo                    Evaluation Script v1.0
echo ============================================================================
echo.

REM ============================================================================
REM  CONFIGURATION
REM ============================================================================

set REAL_PATH=C:\Users\diabo\Desktop\MyDataset\real
set FAKE_PATH=C:\Users\diabo\Desktop\MyDataset\fake
set CHECKPOINT=..\pretrained_weights\fc_weights.pth
set OUTPUT_DIR=vysledky
set BATCH_SIZE=32
set MAX_SAMPLES=0

REM ============================================================================
REM  VALIDATION
REM ============================================================================

echo Configuration:
echo   Real images:  %REAL_PATH%
echo   Fake images:  %FAKE_PATH%
echo   Checkpoint:   %CHECKPOINT%
echo   Output:       %OUTPUT_DIR%
echo   Batch size:   %BATCH_SIZE%
echo.

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

REM ============================================================================
REM  ENVIRONMENT
REM ============================================================================

echo Activating conda environment 'clip'...
call conda activate clip

if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment 'clip'
    echo Create environment with: conda create -n clip python=3.10
    echo Install requirements: pip install torch torchvision scikit-learn tqdm pillow ftfy regex
    pause
    exit /b 1
)

REM Set OpenMP duplicate library workaround
set KMP_DUPLICATE_LIB_OK=TRUE

REM ============================================================================
REM  EXECUTION
REM ============================================================================

echo.
echo Starting evaluation...
echo.

if %MAX_SAMPLES%==0 (
    python evaluate_detector.py ^
        --real_path "%REAL_PATH%" ^
        --fake_path "%FAKE_PATH%" ^
        --checkpoint "%CHECKPOINT%" ^
        --output_dir "%OUTPUT_DIR%" ^
        --batch_size %BATCH_SIZE%
) else (
    python evaluate_detector.py ^
        --real_path "%REAL_PATH%" ^
        --fake_path "%FAKE_PATH%" ^
        --checkpoint "%CHECKPOINT%" ^
        --output_dir "%OUTPUT_DIR%" ^
        --batch_size %BATCH_SIZE% ^
        --max_samples %MAX_SAMPLES%
)

REM ============================================================================
REM  COMPLETION
REM ============================================================================

echo.
echo ============================================================================
echo                        EVALUATION COMPLETE
echo ============================================================================
echo.
echo Results saved to: %OUTPUT_DIR%\MyDataset_clip_results\
echo.
echo Output files:
echo   - metrics.json              (evaluation metrics)
echo.

pause
