"""Updated Gradio UI Components with LoRA Support"""
from __future__ import annotations

import logging
import os
from datetime import datetime

import gradio as gr

from ..data import RyzeDataProcessor
from ..eval import RyzeEvaluator
from ..rl import LoRAManager, RyzeGRPOTrainer, RyzeSFTLoRATrainer

logger = logging.getLogger(__name__)


class DataTabV2:
    """Data processing tab component"""

    def __init__(self):
        self.processor = None

    def create_interface(self) -> gr.Column:
        """Create the data processing interface"""
        with gr.Column():
            gr.Markdown("## 📄 PDF Processing & Dataset Generation")

            with gr.Row():
                pdf_input = gr.File(
                    label="Upload PDF File",
                    file_types=[".pdf"],
                    type="filepath"
                )
                batch_input = gr.File(
                    label="Upload Multiple PDFs (Batch)",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath"
                )

            with gr.Row():
                ocr_lang = gr.Dropdown(
                    choices=["eng", "chi_sim", "eng+chi_sim", "jpn", "kor"],
                    value="eng+chi_sim",
                    label="OCR Language"
                )
                dpi = gr.Slider(
                    minimum=150,
                    maximum=600,
                    value=300,
                    step=50,
                    label="OCR DPI"
                )

            process_btn = gr.Button("🚀 Process PDF(s)", variant="primary")

            with gr.Row():
                ocr_output = gr.Textbox(
                    label="OCR Output (Markdown)",
                    lines=10,
                    max_lines=20
                )
                dataset_output = gr.JSON(
                    label="Generated Dataset Info"
                )

            status_output = gr.Textbox(label="Processing Status", lines=3)

            # Define processing function
            def process_pdfs(single_pdf, batch_pdfs, language, dpi_value):
                try:
                    # Initialize processor
                    config = {
                        'ocr': {
                            'language': language,
                            'dpi': dpi_value
                        }
                    }
                    self.processor = RyzeDataProcessor(config)

                    # Determine which PDFs to process
                    if batch_pdfs:
                        pdf_paths = batch_pdfs
                        results = self.processor.process_batch(pdf_paths)
                        status = f"Processed {len(results)} PDF files in batch mode"
                    elif single_pdf:
                        result = self.processor.process_single_pdf(single_pdf)
                        results = [result]
                        status = f"Processed single PDF: {os.path.basename(single_pdf)}"
                    else:
                        return "", {}, "Error: No PDF file(s) provided"

                    # Extract results for display
                    if results and results[0]['status'] == 'success':
                        # Show OCR output from first file
                        first_result = results[0]
                        markdown_path = first_result['ocr_result']['output_path']

                        with open(markdown_path, 'r', encoding='utf-8') as f:
                            ocr_text = f.read()[:5000]  # Show first 5000 chars

                        # Dataset info
                        dataset_info = {
                            'processed_files': len(results),
                            'output_directory': results[0]['output_dir'],
                            'dataset_stats': results[0]['dataset_result']
                        }

                        return ocr_text, dataset_info, status
                    else:
                        return "", {}, f"Processing failed: {results[0].get('error', 'Unknown error')}"

                except Exception as e:
                    logger.error(f"Processing failed: {str(e)}")
                    return "", {}, f"Error: {str(e)}"

            # Connect the processing function
            process_btn.click(
                fn=process_pdfs,
                inputs=[pdf_input, batch_input, ocr_lang, dpi],
                outputs=[ocr_output, dataset_output, status_output]
            )

        return gr.Column()


class TrainingTabV2:
    """Training tab component with LoRA support"""

    def __init__(self):
        self.sft_trainer = None
        self.grpo_trainer = None

    def create_interface(self) -> gr.Column:
        """Create the training interface"""
        with gr.Column():
            gr.Markdown("## 🚂 Model Training (LoRA + GRPO)")

            with gr.Tab("Stage 1: SFT LoRA Training"):
                gr.Markdown("### 🎯 Supervised Fine-Tuning with LoRA")

                with gr.Row():
                    sft_train_data = gr.Textbox(
                        label="Training Data Path",
                        placeholder="Path to training JSON file"
                    )
                    sft_val_data = gr.Textbox(
                        label="Validation Data Path (Optional)",
                        placeholder="Path to validation JSON file"
                    )

                with gr.Row():
                    sft_base_model = gr.Dropdown(
                        choices=[
                            "microsoft/phi-2",
                            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                            "meta-llama/Llama-2-7b-hf",
                            "Qwen/Qwen-1_8B",
                            "custom"
                        ],
                        value="microsoft/phi-2",
                        label="Base Model"
                    )
                    custom_model_path = gr.Textbox(
                        label="Custom Model Path",
                        placeholder="Path to custom model",
                        visible=False
                    )

                gr.Markdown("#### LoRA Configuration")
                with gr.Row():
                    lora_r = gr.Slider(4, 64, value=16, step=4, label="LoRA r")
                    lora_alpha = gr.Slider(8, 128, value=32, step=8, label="LoRA Alpha")
                    lora_dropout = gr.Number(value=0.1, label="LoRA Dropout")

                with gr.Row():
                    sft_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                    sft_batch_size = gr.Slider(1, 32, value=8, step=1, label="Batch Size")
                    sft_lr = gr.Number(value=3e-4, label="Learning Rate")

                with gr.Row():
                    use_4bit = gr.Checkbox(label="Use 4-bit Quantization", value=False)
                    use_8bit = gr.Checkbox(label="Use 8-bit Quantization", value=False)
                    auto_merge_sft = gr.Checkbox(label="Auto-merge LoRA after training", value=True)

                sft_train_btn = gr.Button("🎯 Start SFT LoRA Training", variant="primary")

                sft_output = gr.Textbox(label="SFT Training Results", lines=10)
                sft_paths_output = gr.JSON(label="Output Paths")

                # Generate GRPO dataset
                gr.Markdown("### 📊 Generate GRPO Dataset from SFT Model")
                with gr.Row():
                    grpo_gen_model_path = gr.Textbox(
                        label="SFT Model Path (merged or LoRA)",
                        placeholder="Path to trained SFT model"
                    )
                    grpo_gen_input_data = gr.Textbox(
                        label="Input Data Path",
                        placeholder="Path to input JSON file"
                    )
                    use_merged_for_gen = gr.Checkbox(
                        label="Use Merged Model",
                        value=True,
                        info="Use merged model instead of LoRA adapter"
                    )

                grpo_gen_btn = gr.Button("🔄 Generate GRPO Dataset")
                grpo_gen_output = gr.Textbox(label="GRPO Dataset Generation Results", lines=5)

            with gr.Tab("Stage 2: GRPO Training"):
                gr.Markdown("### 🚀 Group Relative Policy Optimization with LoRA")

                with gr.Row():
                    grpo_base_model = gr.Textbox(
                        label="Base Model Path (SFT merged model)",
                        placeholder="Path to SFT merged model"
                    )
                    grpo_data_path = gr.Textbox(
                        label="GRPO Dataset Path",
                        placeholder="Path to GRPO training data"
                    )

                gr.Markdown("#### GRPO LoRA Configuration")
                with gr.Row():
                    grpo_lora_r = gr.Slider(4, 32, value=8, step=4, label="GRPO LoRA r")
                    grpo_lora_alpha = gr.Slider(8, 64, value=16, step=8, label="GRPO LoRA Alpha")
                    grpo_lora_dropout = gr.Number(value=0.1, label="GRPO LoRA Dropout")

                with gr.Row():
                    grpo_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                    grpo_batch_size = gr.Slider(1, 16, value=4, step=1, label="Batch Size")
                    grpo_lr = gr.Number(value=5e-5, label="Learning Rate")

                with gr.Row():
                    num_samples = gr.Slider(2, 8, value=4, step=1, label="Samples per Prompt")
                    temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                    kl_coef = gr.Number(value=0.1, label="KL Coefficient")

                with gr.Row():
                    grpo_use_4bit = gr.Checkbox(label="Use 4-bit Quantization", value=False)
                    grpo_use_8bit = gr.Checkbox(label="Use 8-bit Quantization", value=False)
                    auto_merge_grpo = gr.Checkbox(label="Auto-merge LoRA after training", value=True)

                grpo_train_btn = gr.Button("🚀 Start GRPO Training", variant="primary")
                grpo_output = gr.Textbox(label="GRPO Training Results", lines=10)
                grpo_paths_output = gr.JSON(label="Output Paths")

            with gr.Tab("Model Merging"):
                gr.Markdown("### 🔧 Manual LoRA Merging")

                with gr.Row():
                    merge_base_model = gr.Textbox(
                        label="Base Model Path",
                        placeholder="Path to base model"
                    )
                    merge_lora_adapter = gr.Textbox(
                        label="LoRA Adapter Path",
                        placeholder="Path to LoRA adapter"
                    )

                merge_output_path = gr.Textbox(
                    label="Output Path for Merged Model",
                    placeholder="Path to save merged model"
                )

                merge_btn = gr.Button("🔀 Merge LoRA to Base Model")
                merge_output = gr.Textbox(label="Merge Results", lines=5)

                # Sequential merge
                gr.Markdown("### 🔄 Sequential LoRA Merging")
                gr.Markdown("Merge multiple LoRA adapters sequentially (e.g., SFT → GRPO)")

                seq_base_model = gr.Textbox(
                    label="Initial Base Model",
                    placeholder="Path to initial base model"
                )
                lora_paths = gr.Textbox(
                    label="LoRA Adapter Paths (one per line)",
                    lines=3,
                    placeholder="Path to LoRA 1\nPath to LoRA 2\n..."
                )
                seq_output_dir = gr.Textbox(
                    label="Output Directory",
                    placeholder="Directory to save merged models"
                )

                seq_merge_btn = gr.Button("🔄 Sequential Merge")
                seq_merge_output = gr.Textbox(label="Sequential Merge Results", lines=5)

            # Define training functions
            def toggle_custom_model(model_name):
                return gr.Textbox.update(visible=(model_name == "custom"))

            def train_sft_lora(train_path, val_path, base_model, custom_path,
                              r, alpha, dropout, epochs, batch_size, lr,
                              use_4bit_q, use_8bit_q, auto_merge):
                try:
                    model_to_use = custom_path if base_model == "custom" else base_model

                    config = {
                        'base_model_name': model_to_use,
                        'num_epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': lr,
                        'lora_r': r,
                        'lora_alpha': alpha,
                        'lora_dropout': dropout,
                        'use_4bit': use_4bit_q,
                        'use_8bit': use_8bit_q,
                        'auto_merge': auto_merge
                    }

                    self.sft_trainer = RyzeSFTLoRATrainer(config)
                    results = self.sft_trainer.train(train_path, val_path)

                    output_text = f"""SFT LoRA Training Completed!

Base Model: {model_to_use}
Training Loss: {results['training_loss']:.4f}
Total Steps: {results['training_steps']}

LoRA Configuration:
- r: {r}
- alpha: {alpha}
- dropout: {dropout}

Output Directory: {results['run_output_dir']}
"""

                    paths = {
                        'lora_adapter': results['lora_adapter_path'],
                        'merged_model': results.get('merged_model_path', 'Not merged'),
                        'output_dir': results['run_output_dir']
                    }

                    return output_text, paths

                except Exception as e:
                    logger.error(f"SFT LoRA training failed: {str(e)}")
                    return f"Training failed: {str(e)}", {}

            def generate_grpo_dataset(model_path, input_data, use_merged):
                try:
                    if not self.sft_trainer:
                        self.sft_trainer = RyzeSFTLoRATrainer()

                    output_path = f"./grpo_datasets/grpo_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    results = self.sft_trainer.generate_rl_dataset(
                        model_path, input_data, output_path, use_merged_model=use_merged
                    )

                    return f"Generated {results['num_samples']} samples\nSaved to: {results['output_path']}"

                except Exception as e:
                    logger.error(f"GRPO dataset generation failed: {str(e)}")
                    return f"Generation failed: {str(e)}"

            def train_grpo(base_model, data_path, r, alpha, dropout, epochs, batch_size, lr,
                          samples, temp, kl, use_4bit_q, use_8bit_q, auto_merge):
                try:
                    config = {
                        'num_epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': lr,
                        'lora_r': r,
                        'lora_alpha': alpha,
                        'lora_dropout': dropout,
                        'num_samples_per_prompt': samples,
                        'temperature': temp,
                        'kl_coef': kl,
                        'use_4bit': use_4bit_q,
                        'use_8bit': use_8bit_q,
                        'auto_merge': auto_merge
                    }

                    self.grpo_trainer = RyzeGRPOTrainer(config)
                    results = self.grpo_trainer.train(base_model, data_path)

                    # Format training stats
                    final_stats = results['training_stats'][-1] if results['training_stats'] else {}
                    output_text = f"""GRPO Training Completed!

Final Loss: {final_stats.get('loss', 0):.4f}
Final Reward: {final_stats.get('mean_reward', 0):.4f}
Final Normalized Reward: {final_stats.get('mean_normalized_reward', 0):.4f}

LoRA Configuration:
- r: {r}
- alpha: {alpha}
- dropout: {dropout}

Output Directory: {results['run_output_dir']}
"""

                    paths = {
                        'lora_adapter': results['final_lora_path'],
                        'merged_model': results.get('merged_model_path', 'Not merged'),
                        'value_head': results['value_head_path'],
                        'output_dir': results['run_output_dir']
                    }

                    return output_text, paths

                except Exception as e:
                    logger.error(f"GRPO training failed: {str(e)}")
                    return f"Training failed: {str(e)}", {}

            def merge_lora(base_model, lora_adapter, output_path):
                try:
                    merged_path = LoRAManager.merge_lora_to_base(
                        base_model_path=base_model,
                        lora_adapter_path=lora_adapter,
                        output_path=output_path
                    )
                    return f"Successfully merged LoRA!\nMerged model saved to: {merged_path}"
                except Exception as e:
                    logger.error(f"LoRA merge failed: {str(e)}")
                    return f"Merge failed: {str(e)}"

            def sequential_merge(base_model, lora_paths_text, output_dir):
                try:
                    lora_paths = [p.strip() for p in lora_paths_text.split('\n') if p.strip()]

                    final_path = LoRAManager.sequential_merge(
                        base_model_path=base_model,
                        lora_checkpoints=lora_paths,
                        output_base_dir=output_dir
                    )

                    return f"Sequential merge completed!\nFinal model: {final_path}"
                except Exception as e:
                    logger.error(f"Sequential merge failed: {str(e)}")
                    return f"Sequential merge failed: {str(e)}"

            # Connect functions
            sft_base_model.change(
                fn=toggle_custom_model,
                inputs=[sft_base_model],
                outputs=[custom_model_path]
            )

            sft_train_btn.click(
                fn=train_sft_lora,
                inputs=[
                    sft_train_data, sft_val_data, sft_base_model, custom_model_path,
                    lora_r, lora_alpha, lora_dropout, sft_epochs, sft_batch_size, sft_lr,
                    use_4bit, use_8bit, auto_merge_sft
                ],
                outputs=[sft_output, sft_paths_output]
            )

            grpo_gen_btn.click(
                fn=generate_grpo_dataset,
                inputs=[grpo_gen_model_path, grpo_gen_input_data, use_merged_for_gen],
                outputs=[grpo_gen_output]
            )

            grpo_train_btn.click(
                fn=train_grpo,
                inputs=[
                    grpo_base_model, grpo_data_path, grpo_lora_r, grpo_lora_alpha,
                    grpo_lora_dropout, grpo_epochs, grpo_batch_size, grpo_lr,
                    num_samples, temperature, kl_coef, grpo_use_4bit, grpo_use_8bit,
                    auto_merge_grpo
                ],
                outputs=[grpo_output, grpo_paths_output]
            )

            merge_btn.click(
                fn=merge_lora,
                inputs=[merge_base_model, merge_lora_adapter, merge_output_path],
                outputs=[merge_output]
            )

            seq_merge_btn.click(
                fn=sequential_merge,
                inputs=[seq_base_model, lora_paths, seq_output_dir],
                outputs=[seq_merge_output]
            )

        return gr.Column()


class ClusterStatusPanel:
    """Cluster status panel shown when running in distributed mode with Ray."""

    def __init__(self) -> None:
        self._manager = None

    def create_interface(self) -> gr.Column:
        with gr.Column():
            gr.Markdown("## Cluster Status")

            with gr.Row():
                status_output = gr.JSON(label="Cluster Resources")
                active_tasks = gr.JSON(label="Active Tasks")

            refresh_btn = gr.Button("Refresh Status")

            def refresh_status():
                try:

                    if self._manager is None:
                        return {"status": "not connected"}, []
                    health = self._manager.health_check()
                    tasks = self._manager.list_active()
                    return health, tasks
                except Exception as e:
                    return {"error": str(e)}, []

            refresh_btn.click(fn=refresh_status, outputs=[status_output, active_tasks])

        return gr.Column()


class EvaluationTabV2:
    """Evaluation tab component (unchanged)"""

    def __init__(self):
        self.evaluator = None

    def create_interface(self) -> gr.Column:
        """Create the evaluation interface"""
        with gr.Column():
            gr.Markdown("## 📊 Model Evaluation")

            with gr.Tab("Single Model Evaluation"):
                with gr.Row():
                    eval_model_path = gr.Textbox(
                        label="Model Path",
                        placeholder="Path to model to evaluate"
                    )
                    benchmark_name = gr.Dropdown(
                        choices=["general_qa", "summarization", "custom"],
                        value="general_qa",
                        label="Benchmark"
                    )

                with gr.Row():
                    temperature = gr.Slider(0, 1, value=0.7, step=0.1, label="Temperature")
                    do_sample = gr.Checkbox(value=False, label="Do Sample")
                    max_tokens = gr.Slider(50, 1000, value=256, step=50, label="Max New Tokens")

                eval_btn = gr.Button("📈 Evaluate Model", variant="primary")

                with gr.Row():
                    metrics_output = gr.JSON(label="Evaluation Metrics")
                    sample_outputs = gr.Dataframe(
                        label="Sample Outputs",
                        headers=["Instruction", "Expected", "Generated"]
                    )

                eval_status = gr.Textbox(label="Evaluation Status", lines=3)

            with gr.Tab("Model Comparison"):
                models_input = gr.Textbox(
                    label="Model Paths (one per line)",
                    lines=4,
                    placeholder="Path to model 1\nPath to model 2\n..."
                )
                comp_benchmark = gr.Dropdown(
                    choices=["general_qa", "summarization", "custom"],
                    value="general_qa",
                    label="Benchmark for Comparison"
                )

                compare_btn = gr.Button("🔍 Compare Models", variant="primary")
                comparison_output = gr.Dataframe(label="Comparison Results")

            # Define evaluation functions
            def evaluate_single_model(model_path, benchmark, temp, sample, max_tok):
                try:
                    config = {
                        'temperature': temp,
                        'do_sample': sample,
                        'max_new_tokens': max_tok
                    }

                    self.evaluator = RyzeEvaluator(config)
                    results = self.evaluator.evaluate_model(model_path, benchmark)

                    # Extract metrics
                    test_results = results['results_by_split'].get('test', {})
                    metrics = test_results.get('metrics', {})

                    # Prepare sample outputs
                    samples = []
                    detailed_results = test_results.get('detailed_results', [])[:5]
                    for res in detailed_results:
                        samples.append([
                            res['instruction'][:50] + "...",
                            res['expected_output'][:100] + "...",
                            res['model_output'][:100] + "..."
                        ])

                    status = f"Evaluation completed for {os.path.basename(model_path)}"

                    return metrics, samples, status

                except Exception as e:
                    logger.error(f"Evaluation failed: {str(e)}")
                    return {}, [], f"Evaluation failed: {str(e)}"

            def compare_models(model_paths_text, benchmark):
                try:
                    model_paths = [p.strip() for p in model_paths_text.split('\n') if p.strip()]

                    if len(model_paths) < 2:
                        return [], "Please provide at least 2 model paths for comparison"

                    if not self.evaluator:
                        self.evaluator = RyzeEvaluator()

                    comparison_results = self.evaluator.compare_models(model_paths, benchmark)

                    # Format results for dataframe
                    comparison_data = []
                    for model_name, model_results in comparison_results['models'].items():
                        test_metrics = model_results['results_by_split']['test']['metrics']
                        row = {
                            'Model': model_name,
                            'BLEU': f"{test_metrics.get('bleu', 0):.4f}",
                            'ROUGE-1': f"{test_metrics.get('rouge-1', 0):.4f}",
                            'ROUGE-L': f"{test_metrics.get('rouge-l', 0):.4f}",
                            'Exact Match': f"{test_metrics.get('exact_match', 0):.4f}"
                        }
                        comparison_data.append(row)

                    return comparison_data

                except Exception as e:
                    logger.error(f"Model comparison failed: {str(e)}")
                    return [], f"Comparison failed: {str(e)}"

            # Connect functions
            eval_btn.click(
                fn=evaluate_single_model,
                inputs=[eval_model_path, benchmark_name, temperature, do_sample, max_tokens],
                outputs=[metrics_output, sample_outputs, eval_status]
            )

            compare_btn.click(
                fn=compare_models,
                inputs=[models_input, comp_benchmark],
                outputs=[comparison_output]
            )

        return gr.Column()
