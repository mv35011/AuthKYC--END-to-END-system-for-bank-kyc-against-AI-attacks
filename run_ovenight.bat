@echo off
echo ===================================================
echo [System] Starting Defensive KYC Pipeline
echo [System] Environment: Anaconda / Windows 10
echo ===================================================

echo.
echo [1/3] Starting Pipeline Phase 1: Data Extraction...
python data/extractor.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [2/3] Starting Pipeline Phase 2: FTCA Model Training...
python data/train.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [3/3] Starting Pipeline Phase 3: Xception Baseline Training...
python data/train_baseline.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ===================================================
echo [System] ALL OPERATIONS COMPLETED SUCCESSFULLY.
echo ===================================================
pause