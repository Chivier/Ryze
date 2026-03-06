# Ryze Fine-Tune Pipeline

A comprehensive pipeline for fine-tuning LLMs on domain-specific documents. Converts PDF files into training data via OCR, generates QA pairs, and runs two-stage LoRA training (SFT → GRPO).

## Pipeline Overview

```
PDF Library  →  OCR + QA Generation  →  RL Training  →  Fine-tuned Model
```

### Step 1 — Data Processing
- Upload PDF files
- OCR extraction (GLM-OCR, Tesseract, PaddleOCR, EasyOCR, DocTR)
- Automatic QA pair generation via LLM (Qwen3, GPT-4o, Claude, Llama, DeepSeek)

### Step 2 — SFT Training
- Supervised Fine-Tuning with LoRA on generated QA dataset
- Configurable: base model, learning rate, epochs, batch size

### Step 3 — GRPO Training
- Group Relative Policy Optimization on SFT-merged model
- Reward functions: BLEU, ROUGE-L, BERTScore, custom
- LoRA adapter auto-merges to produce final model

## Project Structure

```
Ryze/
├── src/
│   ├── data/       # PDF OCR, QA dataset generation
│   ├── rl/         # SFT and GRPO trainers, LoRA utilities
│   ├── eval/       # Evaluation metrics and benchmarks
│   ├── ui/         # Gradio interface
│   └── figure/     # Figure extraction utilities
├── configs/        # Training configuration files
├── scripts/        # Utility scripts
├── raze_traing_demo/  # Static pipeline demo (index.html)
└── demo/           # Streamlit demos
```

## Installation

```bash
# Install Tesseract OCR
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Install Python dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
./start.sh
```

Or manually:

```bash
python launch_app.py
```

Then open `http://localhost:7860` in your browser.

## Configuration

Edit `configs/default_config_v2.json` to customize OCR settings, training hyperparameters, model paths, and output directories.

## Training Workflow

1. **Data Processing tab** — Upload PDFs, select OCR engine and QA LLM, run processing
2. **Model Training tab** — Configure SFT LoRA, then GRPO LoRA; models auto-merge between stages
3. **Evaluation tab** — Benchmark final model

### LoRA Settings Guide

| Model Size | SFT r / alpha | GRPO r / alpha | Quantization |
|-----------|--------------|----------------|-------------|
| < 3B      | 16 / 32      | 8 / 16         | None        |
| 3B–7B     | 32 / 64      | 16 / 32        | 8-bit       |
| > 7B      | 64 / 128     | 32 / 64        | 4-bit       |

## License

Apache License 2.0
