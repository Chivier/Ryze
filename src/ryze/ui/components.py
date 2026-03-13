"""Gradio UI Components"""
import logging
import os
from datetime import datetime

import gradio as gr

from ..data import RyzeDataProcessor
from ..eval import RyzeEvaluator
from ..rl import RyzeRLTrainer, RyzeSFTTrainer

logger = logging.getLogger(__name__)


class DataTab:
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


class TrainingTab:
    """Training tab component"""

    def __init__(self):
        self.sft_trainer = None
        self.rl_trainer = None

    def create_interface(self) -> gr.Column:
        """Create the training interface"""
        with gr.Column():
            gr.Markdown("## 🚂 Model Training")

            with gr.Tab("SFT Training"):
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
                    sft_model_name = gr.Dropdown(
                        choices=[
                            "microsoft/phi-2",
                            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                            "facebook/opt-125m",
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

                with gr.Row():
                    sft_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                    sft_batch_size = gr.Slider(1, 32, value=8, step=1, label="Batch Size")
                    sft_lr = gr.Number(value=5e-5, label="Learning Rate")

                sft_train_btn = gr.Button("🎯 Start SFT Training", variant="primary")

                sft_output = gr.Textbox(label="SFT Training Results", lines=10)
                sft_model_path_output = gr.Textbox(label="Trained Model Path")

                # RL dataset generation
                gr.Markdown("### Generate RL Dataset from SFT Model")
                with gr.Row():
                    rl_gen_model_path = gr.Textbox(
                        label="SFT Model Path",
                        placeholder="Path to trained SFT model"
                    )
                    rl_gen_input_data = gr.Textbox(
                        label="Input Data Path",
                        placeholder="Path to input JSON file"
                    )

                rl_gen_btn = gr.Button("🔄 Generate RL Dataset")
                rl_gen_output = gr.Textbox(label="RL Dataset Generation Results", lines=5)

            with gr.Tab("RL Training"):
                with gr.Row():
                    rl_model_path = gr.Textbox(
                        label="SFT Model Path",
                        placeholder="Path to SFT model for RL training"
                    )
                    rl_data_path = gr.Textbox(
                        label="RL Dataset Path",
                        placeholder="Path to RL training data"
                    )

                with gr.Row():
                    rl_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                    rl_batch_size = gr.Slider(1, 16, value=4, step=1, label="Batch Size")
                    rl_lr = gr.Number(value=1e-5, label="Learning Rate")

                with gr.Row():
                    ppo_epochs = gr.Slider(1, 10, value=4, step=1, label="PPO Epochs")
                    clip_ratio = gr.Number(value=0.2, label="Clip Ratio")

                rl_train_btn = gr.Button("🚀 Start RL Training", variant="primary")
                rl_output = gr.Textbox(label="RL Training Results", lines=10)
                rl_model_path_output = gr.Textbox(label="RL Model Path")

            # Define training functions
            def toggle_custom_model(model_name):
                return gr.Textbox.update(visible=(model_name == "custom"))

            def train_sft(train_path, val_path, model_name, custom_path, epochs, batch_size, lr):
                try:
                    model_to_use = custom_path if model_name == "custom" else model_name

                    config = {
                        'model_name': model_to_use,
                        'num_epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': lr
                    }

                    self.sft_trainer = RyzeSFTTrainer(config)
                    results = self.sft_trainer.train(train_path, val_path)

                    output_text = f"""SFT Training Completed!

Model: {model_to_use}
Training Loss: {results['training_loss']:.4f}
Total Steps: {results['training_steps']}
Output Directory: {results['run_output_dir']}
Model Saved At: {results['model_save_path']}
"""
                    return output_text, results['model_save_path']

                except Exception as e:
                    logger.error(f"SFT training failed: {str(e)}")
                    return f"Training failed: {str(e)}", ""

            def generate_rl_dataset(sft_model, input_data):
                try:
                    if not self.sft_trainer:
                        self.sft_trainer = RyzeSFTTrainer()

                    output_path = f"./rl_datasets/rl_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    results = self.sft_trainer.generate_rl_dataset(sft_model, input_data, output_path)

                    return f"Generated {results['num_samples']} samples\nSaved to: {results['output_path']}"

                except Exception as e:
                    logger.error(f"RL dataset generation failed: {str(e)}")
                    return f"Generation failed: {str(e)}"

            def train_rl(model_path, data_path, epochs, batch_size, lr, ppo_epochs_val, clip_ratio_val):
                try:
                    config = {
                        'num_epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': lr,
                        'ppo_epochs': ppo_epochs_val,
                        'clip_ratio': clip_ratio_val
                    }

                    self.rl_trainer = RyzeRLTrainer(config)
                    results = self.rl_trainer.train(model_path, data_path)

                    # Format training stats
                    final_stats = results['training_stats'][-1] if results['training_stats'] else {}
                    output_text = f"""RL Training Completed!

Final Loss: {final_stats.get('avg_loss', 0):.4f}
Final Reward: {final_stats.get('avg_reward', 0):.4f}
Output Directory: {results['run_output_dir']}
Model Saved At: {results['final_model_path']}
"""
                    return output_text, results['final_model_path']

                except Exception as e:
                    logger.error(f"RL training failed: {str(e)}")
                    return f"Training failed: {str(e)}", ""

            # Connect functions
            sft_model_name.change(
                fn=toggle_custom_model,
                inputs=[sft_model_name],
                outputs=[custom_model_path]
            )

            sft_train_btn.click(
                fn=train_sft,
                inputs=[
                    sft_train_data, sft_val_data, sft_model_name,
                    custom_model_path, sft_epochs, sft_batch_size, sft_lr
                ],
                outputs=[sft_output, sft_model_path_output]
            )

            rl_gen_btn.click(
                fn=generate_rl_dataset,
                inputs=[rl_gen_model_path, rl_gen_input_data],
                outputs=[rl_gen_output]
            )

            rl_train_btn.click(
                fn=train_rl,
                inputs=[
                    rl_model_path, rl_data_path, rl_epochs,
                    rl_batch_size, rl_lr, ppo_epochs, clip_ratio
                ],
                outputs=[rl_output, rl_model_path_output]
            )

        return gr.Column()


class EvaluationTab:
    """Evaluation tab component"""

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
