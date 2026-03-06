"""Main Gradio Application"""
import gradio as gr
import os
import logging
from typing import Optional, Dict, Any

from .components import DataTab, TrainingTab, EvaluationTab

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RyzeGradioApp:
    """Main Gradio application for Ryze pipeline"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.title = self.config.get('title', 'Ryze-ACL Pipeline')
        self.theme = self.config.get('theme', 'default')

        # Initialize tabs
        self.data_tab = DataTab()
        self.training_tab = TrainingTab()
        self.evaluation_tab = EvaluationTab()

    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface"""
        with gr.Blocks(title=self.title, theme=self.theme) as app:
            # Header
            gr.Markdown("""
            # 🚀 Ryze-ACL Pipeline

            A comprehensive pipeline for PDF OCR, dataset generation, model training (SFT + RL), and evaluation.

            ## Features:
            - 📄 **Data Processing**: Convert PDF to Markdown using OCR and generate SFT datasets
            - 🎯 **SFT Training**: Supervised Fine-Tuning for language models
            - 🔄 **RL Training**: Reinforcement Learning with PPO
            - 📊 **Evaluation**: Comprehensive model evaluation with multiple metrics
            """)

            # Create tabs
            with gr.Tabs():
                with gr.Tab("📄 Data Processing"):
                    self.data_tab.create_interface()

                with gr.Tab("🚂 Model Training"):
                    self.training_tab.create_interface()

                with gr.Tab("📊 Evaluation"):
                    self.evaluation_tab.create_interface()

                with gr.Tab("ℹ️ About"):
                    gr.Markdown("""
                    ## About Ryze-ACL

                    This pipeline integrates three main modules:

                    ### 1. Ryze Data Module
                    - **Input**: PDF files
                    - **Processing**: OCR using Tesseract/PyMuPDF
                    - **Output**: Markdown files and SFT datasets

                    ### 2. Ryze RL Module
                    - **SFT Training**: Fine-tune language models on generated datasets
                    - **RL Training**: Further optimize models using PPO algorithm
                    - **Dataset Generation**: Create RL datasets from SFT outputs

                    ### 3. Ryze Eval Module
                    - **Metrics**: BLEU, ROUGE, Exact Match, Diversity metrics
                    - **Benchmarks**: General QA, Summarization, Custom benchmarks
                    - **Comparison**: Compare multiple models side-by-side

                    ### Quick Start Guide

                    1. **Process PDFs**: Upload PDF files in the Data Processing tab
                    2. **Train SFT Model**: Use generated datasets to train an SFT model
                    3. **Generate RL Dataset**: Create RL training data from SFT model
                    4. **Train with RL**: Further optimize the model using RL
                    5. **Evaluate**: Test model performance on benchmarks

                    ### Requirements
                    - CUDA-capable GPU recommended for training
                    - Tesseract OCR installed for PDF processing
                    - Sufficient disk space for model checkpoints

                    ### Configuration
                    All outputs are saved in organized directories:
                    - `./output/` - OCR and dataset outputs
                    - `./sft_outputs/` - SFT training results
                    - `./rl_outputs/` - RL training results
                    - `./eval_outputs/` - Evaluation results
                    """)

                with gr.Tab("⚙️ Settings"):
                    gr.Markdown("## Pipeline Settings")

                    with gr.Row():
                        output_base = gr.Textbox(
                            label="Base Output Directory",
                            value="./output",
                            info="Root directory for all outputs"
                        )
                        log_level = gr.Dropdown(
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                            value="INFO",
                            label="Logging Level"
                        )

                    with gr.Row():
                        gpu_device = gr.Number(
                            value=0,
                            label="GPU Device ID",
                            info="CUDA device to use (-1 for CPU)"
                        )
                        cache_dir = gr.Textbox(
                            label="Model Cache Directory",
                            value="~/.cache/huggingface",
                            info="Directory for caching downloaded models"
                        )

                    save_settings_btn = gr.Button("💾 Save Settings")
                    settings_status = gr.Textbox(label="Status", lines=2)

                    def save_settings(output_dir, log_lvl, gpu_id, cache):
                        try:
                            # Update logging level
                            logging.getLogger().setLevel(getattr(logging, log_lvl))

                            # Create directories if they don't exist
                            os.makedirs(output_dir, exist_ok=True)
                            os.makedirs(os.path.expanduser(cache), exist_ok=True)

                            # Set environment variables
                            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                            os.environ['HF_HOME'] = os.path.expanduser(cache)

                            return "Settings saved successfully!"

                        except Exception as e:
                            return f"Failed to save settings: {str(e)}"

                    save_settings_btn.click(
                        fn=save_settings,
                        inputs=[output_base, log_level, gpu_device, cache_dir],
                        outputs=[settings_status]
                    )

            # Footer
            gr.Markdown("""
            ---
            **Ryze-ACL Pipeline** | Version 0.1.0 | [GitHub](https://github.com/yourusername/Ryze-ACL)
            """)

        return app

    def launch(self, **kwargs):
        """Launch the Gradio app"""
        app = self.create_interface()

        # Default launch parameters
        launch_params = {
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'share': False,
            'inbrowser': True
        }

        # Update with any provided parameters
        launch_params.update(kwargs)

        logger.info(f"Launching Ryze-ACL Pipeline on port {launch_params['server_port']}")
        app.launch(**launch_params)


def main():
    """Main entry point"""
    app = RyzeGradioApp()
    app.launch()


if __name__ == "__main__":
    main()