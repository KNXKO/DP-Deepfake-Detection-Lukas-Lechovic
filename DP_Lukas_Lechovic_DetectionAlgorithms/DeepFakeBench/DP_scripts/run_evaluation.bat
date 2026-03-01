@echo off
REM ============================================================================
REM  PRETRAINED DEEPFAKE DETECTOR EVALUATION
REM ============================================================================
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo   PRETRAINED DEEPFAKE DETECTOR EVALUATION
echo ============================================================================
echo.
echo   This script tests selected pretrained model on your data.
echo   Results (metrics, graphs) will be saved to evaluation_results/
echo.
echo ============================================================================
echo.
echo Available detectors:
echo.
echo   [1] Xception      - CNN model based on Xception architecture
echo   [2] Meso4         - Simpler model for face manipulation detection
echo.
echo ============================================================================
echo.

set /p CHOICE=Select detector (1 or 2):

if "%CHOICE%"=="1" (
    set DETECTOR=xception
    set WEIGHTS=..\training\weights\xception_best.pth
    set CONFIG=..\training\config\detector\xception.yaml
    set DETECTOR_NAME=Xception
)
if "%CHOICE%"=="2" (
    set DETECTOR=meso4
    set WEIGHTS=..\training\weights\meso4_best.pth
    set CONFIG=..\training\config\detector\meso4.yaml
    set DETECTOR_NAME=Meso4
)

if not defined DETECTOR (
    echo.
    echo [ERROR] Invalid choice. Select 1 or 2.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo   CONFIGURATION
echo ============================================================================
echo.
echo   Model:     %DETECTOR_NAME%
echo   Weights:   %WEIGHTS%
echo   Config:    %CONFIG%
echo.
echo ============================================================================
echo.

REM Check file existence
if not exist "%WEIGHTS%" (
    echo [ERROR] Weights file does not exist: %WEIGHTS%
    echo.
    echo Make sure you have downloaded pretrained weights.
    pause
    exit /b 1
)

if not exist "%CONFIG%" (
    echo [ERROR] Config file does not exist: %CONFIG%
    pause
    exit /b 1
)

echo Press any key to start evaluation...
pause > nul

echo.
echo Activating conda environment deepfakebench...
call conda activate deepfakebench

if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment 'deepfakebench'
    echo Check if environment exists: conda env list
    pause
    exit /b 1
)

echo.
echo Starting evaluation...
echo.

REM Change to project root for correct relative paths
cd ..

python DP_scripts\evaluate_detector.py ^
    --detector_path training\config\detector\%DETECTOR%.yaml ^
    --test_dataset MyDataset_full ^
    --weights_path training\weights\%DETECTOR%_best.pth ^
    --checkpoint_dir DP_scripts\evaluation_results\%DETECTOR%

REM Return to DP_scripts
cd DP_scripts

echo.
echo ============================================================================
echo   RESULTS FOR %DETECTOR_NAME%
echo ============================================================================
echo.

set RESULTS_FILE=evaluation_results\%DETECTOR%\MyDataset_full_results\metrics.json

if exist "%RESULTS_FILE%" (
    echo Contents of metrics.json:
    echo.
    type "%RESULTS_FILE%"
    echo.
    echo.
    echo Complete results in: evaluation_results\%DETECTOR%\MyDataset_full_results\
    echo   - metrics.json          ^(numeric metrics^)
    echo   - roc_curve.png         ^(ROC curve^)
    echo   - confusion_matrix.png  ^(confusion matrix^)
    echo   - precision_recall_curve.png
) else (
    echo [WARNING] Results file not found.
    echo Check console output.
)

echo.
echo ============================================================================
pause
