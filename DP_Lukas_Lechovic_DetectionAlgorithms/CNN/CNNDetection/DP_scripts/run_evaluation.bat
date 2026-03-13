@echo off
REM ============================================================
REM CNNDetection - Dataset Evaluation Script
REM ============================================================
REM This script runs the deepfake detector on a specified dataset
REM and saves the results to the vysledky folder.
REM
REM Prerequisites:
REM   - Anaconda with 'cnndetection' environment
REM   - Pre-trained weights in ../weights/ folder
REM   - Dataset with real/ and fake/ subfolders
REM
REM Usage:
REM   1. Edit configuration below
REM   2. Run this script: run_evaluation.bat
REM ============================================================

REM === CONFIGURATION ===
SET DATASET_PATH=C:\Users\diabo\Desktop\FFHQ-FaceFusion-10k
SET MODEL_PATH=..\weights\blur_jpg_prob0.5.pth
SET CONDA_ENV=cnndetection
SET THRESHOLD=0.5
SET MAX_SAMPLES=

REM === VALIDATE INPUTS ===
echo ======================================================================
echo                  CNNDetection - Evaluation
echo ======================================================================

IF NOT EXIST "%DATASET_PATH%" (
    echo ERROR: Dataset path does not exist: %DATASET_PATH%
    pause
    exit /b 1
)

IF NOT EXIST "%MODEL_PATH%" (
    echo ERROR: Model file does not exist: %MODEL_PATH%
    pause
    exit /b 1
)

REM Check dataset subfolders
SET FOUND_REAL=0
SET FOUND_FAKE=0
IF EXIST "%DATASET_PATH%\real" SET FOUND_REAL=1
IF EXIST "%DATASET_PATH%\0_real" SET FOUND_REAL=1
IF EXIST "%DATASET_PATH%\Real" SET FOUND_REAL=1
IF EXIST "%DATASET_PATH%\fake" SET FOUND_FAKE=1
IF EXIST "%DATASET_PATH%\1_fake" SET FOUND_FAKE=1
IF EXIST "%DATASET_PATH%\Fake" SET FOUND_FAKE=1

IF "%FOUND_REAL%"=="0" (
    echo ERROR: No real/ or 0_real/ folder found in %DATASET_PATH%
    pause
    exit /b 1
)
IF "%FOUND_FAKE%"=="0" (
    echo ERROR: No fake/ or 1_fake/ folder found in %DATASET_PATH%
    pause
    exit /b 1
)

REM === ACTIVATE ENVIRONMENT ===
echo Activating conda environment: %CONDA_ENV%
call conda activate %CONDA_ENV%

REM === BUILD COMMAND ===
SET CMD=python evaluate_detector.py -d "%DATASET_PATH%" -m "%MODEL_PATH%" --threshold %THRESHOLD%

IF NOT "%MAX_SAMPLES%"=="" (
    SET CMD=%CMD% --max_samples %MAX_SAMPLES%
    echo [QUICK TEST MODE] Max samples per class: %MAX_SAMPLES%
)

REM === RUN EVALUATION ===
echo.
echo Starting evaluation...
echo Dataset: %DATASET_PATH%
echo Model:   %MODEL_PATH%
echo Threshold: %THRESHOLD%
echo.

%CMD%

echo.
echo ======================================================================
echo Evaluation complete. Results saved in vysledky folder.
echo ======================================================================
pause
