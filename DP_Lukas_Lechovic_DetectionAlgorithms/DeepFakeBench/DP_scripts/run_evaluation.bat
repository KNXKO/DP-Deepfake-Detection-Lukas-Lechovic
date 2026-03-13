@echo off
REM ============================================================================
REM  XCEPTION DEEPFAKE DETECTOR EVALUATION
REM ============================================================================
REM  Author: Diploma Thesis
REM  Description: Evaluate Xception detector on FFHQ-FaceFusion-10k dataset
REM ============================================================================

setlocal enabledelayedexpansion

REM ============================================================================
REM  CONFIGURATION - Edit these variables as needed
REM ============================================================================

set DATASET_PATH=C:\Users\diabo\Desktop\FFHQ-FaceFusion-10k
set DATASET_NAME=FFHQ-FaceFusion-10k_full
set WEIGHTS_PATH=..\training\weights\xception_best.pth
set DETECTOR_CONFIG=..\training\config\detector\xception.yaml
set BATCH_SIZE=32
set CONDA_ENV=deepfakebench

REM Optional: limit samples for quick test (set to 0 for all samples)
set MAX_SAMPLES=0

REM ============================================================================

echo.
echo ============================================================================
echo   XCEPTION DEEPFAKE DETECTOR EVALUATION
echo ============================================================================
echo.
echo   Configuration:
echo     Dataset:      %DATASET_PATH%
echo     Dataset name: %DATASET_NAME%
echo     Weights:      %WEIGHTS_PATH%
echo     Config:       %DETECTOR_CONFIG%
echo     Batch size:   %BATCH_SIZE%
if %MAX_SAMPLES% GTR 0 (
    echo     Max samples:  %MAX_SAMPLES% ^(quick test mode^)
) else (
    echo     Max samples:  all
)
echo.
echo ============================================================================
echo.

REM ---- Validate dataset path ----
if not exist "%DATASET_PATH%" (
    echo [ERROR] Dataset directory does not exist: %DATASET_PATH%
    pause
    exit /b 1
)

REM Check for real/fake or 0_real/1_fake folders
set REAL_FOUND=0
if exist "%DATASET_PATH%\real" set REAL_FOUND=1
if exist "%DATASET_PATH%\0_real" set REAL_FOUND=1

set FAKE_FOUND=0
if exist "%DATASET_PATH%\fake" set FAKE_FOUND=1
if exist "%DATASET_PATH%\1_fake" set FAKE_FOUND=1

if %REAL_FOUND%==0 (
    echo [ERROR] No 'real' or '0_real' folder found in dataset path
    pause
    exit /b 1
)
if %FAKE_FOUND%==0 (
    echo [ERROR] No 'fake' or '1_fake' folder found in dataset path
    pause
    exit /b 1
)

REM ---- Validate weights file ----
if not exist "%WEIGHTS_PATH%" (
    echo [ERROR] Weights file does not exist: %WEIGHTS_PATH%
    echo.
    echo Make sure you have trained or downloaded Xception weights.
    pause
    exit /b 1
)

REM ---- Validate config file ----
if not exist "%DETECTOR_CONFIG%" (
    echo [ERROR] Config file does not exist: %DETECTOR_CONFIG%
    pause
    exit /b 1
)

echo Press any key to start evaluation...
pause > nul

REM ---- Activate conda environment ----
echo.
echo Activating conda environment %CONDA_ENV%...
call conda activate %CONDA_ENV%

if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment '%CONDA_ENV%'
    echo Check if environment exists: conda env list
    pause
    exit /b 1
)

REM ---- Change to project root ----
cd ..

REM ---- Step 1: Prepare dataset JSON (if not exists) ----
set JSON_PATH=preprocessing\dataset_json\%DATASET_NAME%.json
if not exist "%JSON_PATH%" (
    echo.
    echo [i] Dataset JSON not found, generating...
    python DP_scripts\prepare_dataset.py ^
        --dataset_path "%DATASET_PATH%" ^
        --dataset_name %DATASET_NAME%

    if errorlevel 1 (
        echo [ERROR] Failed to generate dataset JSON
        cd DP_scripts
        pause
        exit /b 1
    )
    echo.
) else (
    echo [i] Dataset JSON already exists: %JSON_PATH%
)

REM ---- Step 2: Run evaluation ----
echo.
echo Starting Xception evaluation...
echo.

if %MAX_SAMPLES% GTR 0 (
    python DP_scripts\evaluate_detector.py ^
        --detector_path training\config\detector\xception.yaml ^
        --test_dataset %DATASET_NAME% ^
        --weights_path training\weights\xception_best.pth ^
        --max_samples %MAX_SAMPLES%
) else (
    python DP_scripts\evaluate_detector.py ^
        --detector_path training\config\detector\xception.yaml ^
        --test_dataset %DATASET_NAME% ^
        --weights_path training\weights\xception_best.pth
)

REM ---- Return to DP_scripts ----
cd DP_scripts

echo.
echo ============================================================================
echo   EVALUATION COMPLETE
echo ============================================================================
echo.
echo   Results saved in: DP_scripts\vysledky\
echo.
echo   To analyze results at different thresholds:
echo     python analyze_results.py --results_dir vysledky\^<folder_name^>
echo     python analyze_results.py --results_dir vysledky\^<folder_name^> --threshold 0.6
echo.
echo ============================================================================
pause
