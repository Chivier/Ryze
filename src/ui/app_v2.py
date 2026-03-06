"""Main Gradio Application with LoRA Support"""
import gradio as gr
import os
import logging
from typing import Optional, Dict, Any

from .components_v2 import DataTabV2, TrainingTabV2, EvaluationTabV2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RyzeGradioAppV2:
    """Main Gradio application for Ryze pipeline with LoRA support"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.title = self.config.get('title', 'Ryze-ACL Pipeline v2')
        self.theme = self.config.get('theme', 'default')

        # Initialize tabs
        self.data_tab = DataTabV2()
        self.training_tab = TrainingTabV2()
        self.evaluation_tab = EvaluationTabV2()

    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface"""
        with gr.Blocks(title=self.title, theme=self.theme) as app:
            # Header
            gr.Markdown("""
            # 🚀 Ryze-ACL Pipeline v2

            A comprehensive pipeline for PDF OCR, dataset generation, and two-stage LoRA training (SFT + GRPO).

            ## 🔥 New Features:
            - **LoRA Training**: Efficient parameter-efficient fine-tuning
            - **GRPO**: Group Relative Policy Optimization for better performance
            - **Stage-wise Merging**: SFT and GRPO LoRA adapters are merged sequentially
            - **Quantization Support**: 4-bit and 8-bit training for reduced memory usage

            ## 📚 Training Workflow:
            1. **Data Processing**: PDF → OCR → Markdown → SFT Dataset
            2. **Stage 1 - SFT LoRA**: Supervised fine-tuning with LoRA → Merge to base
            3. **Stage 2 - GRPO LoRA**: Policy optimization on merged model → Final merge
            4. **Evaluation**: Test the final model on benchmarks
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
                    ## About Ryze-ACL v2

                    This updated pipeline implements a two-stage training approach with LoRA:

                    ### 🎯 Stage 1: SFT LoRA Training
                    - **Purpose**: Initial supervised fine-tuning on your dataset
                    - **Method**: LoRA (Low-Rank Adaptation) for efficient training
                    - **Output**: LoRA adapter + optional merged model

                    ### 🚀 Stage 2: GRPO Training
                    - **Purpose**: Further optimization using group-relative rewards
                    - **Method**: GRPO with LoRA on the SFT-merged model
                    - **Output**: Final LoRA adapter + merged model

                    ### 🔧 Key Features

                    #### LoRA Configuration
                    - **r**: Rank of adaptation (higher = more parameters)
                    - **alpha**: Scaling factor for LoRA
                    - **dropout**: Regularization for LoRA layers

                    #### GRPO Settings
                    - **Samples per Prompt**: Generate multiple responses for comparison
                    - **Temperature**: Control randomness in generation
                    - **KL Coefficient**: Balance between improvement and staying close to SFT

                    ### 📊 Memory Optimization
                    - **4-bit Quantization**: ~4x memory reduction
                    - **8-bit Quantization**: ~2x memory reduction
                    - **Gradient Checkpointing**: Trade computation for memory

                    ### 🔄 Merging Strategy
                    1. Train SFT LoRA on base model
                    2. Merge SFT LoRA → Create SFT-merged model
                    3. Train GRPO LoRA on SFT-merged model
                    4. Merge GRPO LoRA → Create final model

                    ### 📁 Output Structure
                    ```
                    sft_lora_outputs/
                    ├── sft_lora_<timestamp>/
                    │   ├── lora_adapter/     # LoRA weights
                    │   ├── merged_model/     # Merged model (optional)
                    │   └── training_results.json

                    grpo_outputs/
                    ├── grpo_lora_<timestamp>/
                    │   ├── final_lora_adapter/
                    │   ├── merged_model/     # Final model
                    │   ├── value_head.pt     # Value network
                    │   └── training_results.json
                    ```

                    ### 🎓 Recommended Settings

                    **For Small Models (< 3B params):**
                    - SFT LoRA: r=16, alpha=32
                    - GRPO LoRA: r=8, alpha=16
                    - No quantization needed

                    **For Medium Models (3B-7B params):**
                    - SFT LoRA: r=32, alpha=64
                    - GRPO LoRA: r=16, alpha=32
                    - Consider 8-bit quantization

                    **For Large Models (> 7B params):**
                    - SFT LoRA: r=64, alpha=128
                    - GRPO LoRA: r=32, alpha=64
                    - Use 4-bit quantization
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

                    # PEFT library settings
                    gr.Markdown("### LoRA Library Settings")
                    with gr.Row():
                        peft_backend = gr.Dropdown(
                            choices=["peft", "unsloth"],
                            value="peft",
                            label="LoRA Backend",
                            info="Choose LoRA implementation"
                        )
                        enable_flash_attn = gr.Checkbox(
                            label="Enable Flash Attention",
                            value=False,
                            info="Requires flash-attn package"
                        )

                    save_settings_btn = gr.Button("💾 Save Settings")
                    settings_status = gr.Textbox(label="Status", lines=2)

                    def save_settings(output_dir, log_lvl, gpu_id, cache, backend, flash_attn):
                        try:
                            # Update logging level
                            logging.getLogger().setLevel(getattr(logging, log_lvl))

                            # Create directories if they don't exist
                            os.makedirs(output_dir, exist_ok=True)
                            os.makedirs(os.path.expanduser(cache), exist_ok=True)
                            os.makedirs("./sft_lora_outputs", exist_ok=True)
                            os.makedirs("./grpo_outputs", exist_ok=True)
                            os.makedirs("./grpo_datasets", exist_ok=True)

                            # Set environment variables
                            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                            os.environ['HF_HOME'] = os.path.expanduser(cache)

                            # Set backend preferences
                            os.environ['PEFT_BACKEND'] = backend
                            if flash_attn:
                                os.environ['ENABLE_FLASH_ATTN'] = "1"

                            return "Settings saved successfully!"

                        except Exception as e:
                            return f"Failed to save settings: {str(e)}"

                    save_settings_btn.click(
                        fn=save_settings,
                        inputs=[output_base, log_level, gpu_device, cache_dir, peft_backend, enable_flash_attn],
                        outputs=[settings_status]
                    )

            # Footer
            gr.Markdown("""
            ---
            **Ryze-ACL Pipeline v2** | Version 0.2.0 | [GitHub](https://github.com/yourusername/Ryze-ACL)

            Built with ❤️ using Hugging Face Transformers, PEFT, and Gradio
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

        logger.info(f"Launching Ryze-ACL Pipeline v2 on port {launch_params['server_port']}")
        app.launch(**launch_params)


def main():
    """Main entry point"""
    app = RyzeGradioAppV2()
    app.launch()


if __name__ == "__main__":
    main()