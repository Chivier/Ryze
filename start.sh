#!/bin/bash
# One-click start script for Ryze-ACL Pipeline v2 with LoRA support

echo "=============================================="
echo "   Starting Ryze-ACL Pipeline v2"
echo "   SFT + GRPO with LoRA Training"
echo "=============================================="

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "It's recommended to use a virtual environment."
    echo ""
    read -p "Do you want to create one now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        echo "✅ Virtual environment created and activated."
    fi
fi

# Install dependencies if requirements_v2.txt exists and packages not installed
if [ -f "requirements_v2.txt" ]; then
    echo "📦 Checking dependencies..."
    if ! python -c "import peft" 2>/dev/null; then
        echo "Installing required packages..."
        pip install -r requirements_v2.txt
        echo "✅ Dependencies installed."
    else
        echo "✅ Dependencies already installed."
    fi
fi

# Check for PEFT library
echo ""
echo "🔍 Checking LoRA backend..."
if python -c "import peft; print(f'✅ PEFT version: {peft.__version__}')" 2>/dev/null; then
    :
else
    echo "⚠️  PEFT not found. Installing..."
    pip install peft
fi

# Check for optional optimizations
echo ""
echo "📊 Checking optional optimizations:"

# Bitsandbytes for quantization
if python -c "import bitsandbytes" 2>/dev/null; then
    echo "✅ Bitsandbytes: Available (4-bit/8-bit quantization supported)"
else
    echo "⚠️  Bitsandbytes: Not installed (required for quantization)"
    echo "   Install with: pip install bitsandbytes"
fi

# Flash Attention
if python -c "import flash_attn" 2>/dev/null; then
    echo "✅ Flash Attention: Available (faster training)"
else
    echo "ℹ️  Flash Attention: Not installed (optional, for faster training)"
fi

# Check for Tesseract OCR
if ! command -v tesseract &> /dev/null; then
    echo ""
    echo "⚠️  Warning: Tesseract OCR not found."
    echo "Please install Tesseract for PDF OCR functionality:"
    echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim"
    echo "  macOS: brew install tesseract"
    echo "  Windows: Download from https://github.com/tesseract-ocr/tesseract"
    echo ""
fi

# Check for GPU
echo ""
python -c "import torch; print(f'🖥️  GPU Available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'🎮 GPU: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; print(f'💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
else
    echo "⚠️  No GPU detected. Training will be slow on CPU."
fi

echo ""
echo "🚀 Launching Ryze-ACL Pipeline v2..."
echo "📚 Training workflow: PDF → OCR → SFT LoRA → Merge → GRPO → Final Model"
echo ""

# Launch with arguments passed to the script
python launch_app_v2.py "$@"