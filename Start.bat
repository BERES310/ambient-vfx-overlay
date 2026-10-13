@echo off
rem One-click launcher: creates the local venv if missing, installs requirements once,
rem then opens the VFX Overlay app. Everything (venv, pip cache, AI models) stays inside this folder.
setlocal
cd /d "%~dp0"

set "PIP_CACHE_DIR=%~dp0models\pip_cache"
set "HF_HOME=%~dp0models\huggingface"
set "TRANSFORMERS_CACHE=%~dp0models\huggingface"
set "TORCH_HOME=%~dp0models\torch"

if not exist "venv\Scripts\python.exe" (
    echo Creating the local Python environment ^(first run only^)...
    where py >nul 2>nul && (py -3 -m venv venv) || (python -m venv venv)
    if errorlevel 1 (
        echo.
        echo Python 3 was not found. Please install Python 3.10 or newer from python.org and run this file again.
        pause
        exit /b 1
    )
)

if not exist "venv\.requirements_installed" (
    echo Installing the required libraries ^(first run only, this can take several minutes^)...
    "venv\Scripts\python.exe" -m pip install --upgrade pip
    rem If an NVIDIA GPU is present, install the CUDA build of PyTorch first so the AI steps run fast.
    where nvidia-smi >nul 2>nul && (
        echo NVIDIA GPU detected - installing the CUDA build of PyTorch...
        "venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    )
    "venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Installing the libraries failed. Check your internet connection and run this file again.
        pause
        exit /b 1
    )
    echo done> "venv\.requirements_installed"
)

echo Starting VFX Overlay ^(AI models download automatically into the models folder the first time they are needed^)...
"venv\Scripts\python.exe" vfx_gui.py
if errorlevel 1 (
    echo.
    echo The app closed with an error. Details are in output\gui_error.log
    pause
)
endlocal
