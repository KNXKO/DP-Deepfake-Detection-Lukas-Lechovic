@echo off
setlocal enabledelayedexpansion

REM Configuration
set count=0
set max_count=10
set skip_existing=Y

REM Ask user for configuration
echo ================================================
echo FaceFusion - Male Face Processing (Optimized)
echo ================================================
echo.
set /p max_count="How many images to process? [default: 10]: "
if "!max_count!"=="" set max_count=10
echo.
set /p skip_existing="Skip already processed files? (Y/N) [default: Y]: "
if "!skip_existing!"=="" set skip_existing=Y
echo.

REM Convert to uppercase for comparison
if /i "!skip_existing!"=="Y" (
    echo [MODE] Skipping existing files
) else (
    echo [MODE] Overwriting existing files
)
echo.

cd /d "C:\pinokio\api\facefusion-pinokio.git\facefusion"
call conda activate C:\pinokio\api\facefusion-pinokio.git\.env

echo Processing male faces with optimized parameters...
echo Target: !max_count! images
echo ================================================
echo.

for %%f in ("C:\Users\diabo\Desktop\MyDataset\real\*.png" "C:\Users\diabo\Desktop\MyDataset\real\*.jpg") do (
    if !count! LSS %max_count% (
        set "should_process=1"

        REM Check if we should skip existing files
        if /i "!skip_existing!"=="Y" (
            if exist "C:\Users\diabo\Desktop\MyDataset\fake\fake_%%~nxf" (
                echo [SKIP] fake_%%~nxf already exists, skipping...
                set "should_process=0"
            )
        )

        if "!should_process!"=="1" (
            set /a count+=1
            echo ================================================
            echo Processing [!count!/%max_count%]: %%~nxf
            echo ================================================

            python facefusion.py headless-run -t "%%f" -o "C:\Users\diabo\Desktop\MyDataset\fake\fake_%%~nxf" --processors deep_swapper face_enhancer expression_restorer --deep-swapper-model iperov/elon_musk_224 --deep-swapper-morph 80 --face-swapper-model hyperswap_1c_256 --face-swapper-pixel-boost 1024x1024 --face-enhancer-model gpen_bfr_1024 --face-enhancer-blend 80 --face-enhancer-weight 1.0 --expression-restorer-model live_portrait --expression-restorer-factor 80 --face-selector-mode one --face-selector-order large-small --face-selector-age-start 0 --face-selector-age-end 100 --face-occluder-model xseg_3 --face-parser-model bisenet_resnet_34 --face-mask-types box occlusion area --face-mask-blur 0.3 --face-mask-padding 5 5 5 5 --face-detector-model yolo_face --face-detector-size 640x640 --face-detector-angles 0 --face-detector-score 0.5 --face-landmarker-model 2dfan4 --face-landmarker-score 0.4 --execution-providers cuda --execution-thread-count 16 --execution-queue-count 1 --output-image-quality 90 --output-image-resolution 1024x1024

            if errorlevel 1 (
                echo [ERROR] Failed to process: fake_%%~nxf
            ) else (
                echo [SUCCESS] Completed: fake_%%~nxf
            )
            echo.
        )
    )
)

echo ================================================
echo DONE! Processed !count! male faces.
echo ================================================
pause