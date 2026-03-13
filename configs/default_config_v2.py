"""Create default configuration for v2"""
import json


def create_default_config(output_path: str):
    """Create default v2 configuration"""
    config = {
        "data_processing": {
            "ocr": {
                "language": "eng+chi_sim",
                "dpi": 300,
                "use_gpu": False
            },
            "dataset": {
                "min_text_length": 50,
                "max_text_length": 2048,
                "instruction_templates": [
                    "Please summarize the following text:",
                    "Extract the key points from this document:",
                    "What is the main topic of this text?",
                    "Identify the important information in this passage:",
                    "Provide a brief overview of the following content:"
                ]
            },
            "output_base": "./output"
        },
        "training": {
            "sft": {
                "base_model_name": "microsoft/phi-2",
                "batch_size": 8,
                "learning_rate": 3e-4,
                "num_epochs": 3,
                "max_length": 2048,
                "gradient_accumulation_steps": 4,
                "warmup_ratio": 0.03,
                "weight_decay": 0.001,
                "output_dir": "./sft_lora_outputs",
                "lora": {
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                    "target_modules": None,
                    "use_4bit": False,
                    "use_8bit": False
                },
                "auto_merge": True
            },
            "rl": {
                "batch_size": 4,
                "micro_batch_size": 1,
                "learning_rate": 5e-5,
                "num_epochs": 3,
                "max_length": 1024,
                "max_new_tokens": 256,
                "output_dir": "./grpo_outputs",
                "grpo": {
                    "num_samples_per_prompt": 4,
                    "temperature": 0.8,
                    "kl_coef": 0.1,
                    "clip_range": 0.2,
                    "value_clip_range": 0.2,
                    "grpo_epochs": 4
                },
                "lora": {
                    "r": 8,
                    "alpha": 16,
                    "dropout": 0.1,
                    "target_modules": None,
                    "use_4bit": False,
                    "use_8bit": False
                },
                "auto_merge": True
            }
        },
        "evaluation": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "do_sample": False,
            "top_p": 0.95,
            "output_dir": "./eval_outputs",
            "benchmarks_dir": "./benchmarks"
        },
        "ui": {
            "title": "Ryze-ACL Pipeline v2",
            "theme": "default",
            "server": {
                "host": "0.0.0.0",
                "port": 7860,
                "share": False,
                "debug": False
            }
        },
        "cluster": {
            "mode": "local",
            "pylet_head_url": "http://localhost:8000",
            "timeout_s": 300,
            "max_retries": 3
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    create_default_config("default_config_v2.json")