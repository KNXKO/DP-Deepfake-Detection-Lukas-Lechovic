@echo off
REM ============================================================
REM CNNDetection - Dataset Evaluation Script
REM ============================================================
REM This script runs the deepfake detector on a specified dataset
REM and saves the results to the vysledky folder.
REM
REM Prerequisites:
REM   - Anaconda with 'glhf' environment
REM   - Pre-trained weights in ../weights/ folder
REM   - Dataset with real/ and fake/ subfolders
REM
REM Usage:
REM   1. Edit DATASET_PATH below to point to your dataset
REM   2. Run this script: run_evaluation.bat
REM ============================================================

REM === CONFIGURATION ===
SET DATASET_PATH=C:\Users\diabo\Desktop\MyDataset
SET MODEL_PATH=..\weights\blur_jpg_prob0.5.pth
SET CONDA_ENV=glhf

REM === ACTIVATE ENVIRONMENT ===
echo Activating conda environment: %CONDA_ENV%
call conda activate %CONDA_ENV%

REM === RUN EVALUATION ===
echo.
echo Starting evaluation...
echo Dataset: %DATASET_PATH%
echo Model: %MODEL_PATH%
echo.

python evaluate_detector.py -d %DATASET_PATH% -m %MODEL_PATH%

echo.
echo Evaluation complete. Results saved in vysledky folder.
pause
