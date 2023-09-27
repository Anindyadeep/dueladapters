import os
from dataclasses import dataclass, field
from typing import List
import torch
from transformers import BitsAndBytesConfig


### prompts ###
# Format this instruction template according to the dataset.

INSTRUCION_FINETUNING_PROMPT = """
### Human: Summarize the below customer complaint by choosing any of the two possible outcome/class:
1. Bureau discrepancy
2. Improper Bureau Reporting

{text}

### Assistant: {label}
"""

current_script_dir = os.path.dirname(
    os.path.abspath(__file__)
)

# Navigate up on directory and create a folder inside the root
project_dir = os.path.dirname(current_script_dir)
log_dir = os.path.join(project_dir, "Logs")

class GeneralConfig:
    project_dir: str = project_dir
    log_dir: str = log_dir


# assumption: the data must contain these columns
# status: [train, test]
# complaint: str
# issue: str

### Model Quantization config ###

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

### Dataset config ###
# For pre-fine-training

@dataclass
class PreFineTuneDatasetConfig:
    df_path: str = 'Data/pre_finetuning.csv'
    feature_col: str = 'complaint'
    label_col: str = 'issue'
    status_col: str = 'status'
    model_id: str = "tiiuae/falcon-7b"
    context_length: str = 512
    
    # === hf save directory ====
    hf_dir: str = "Data/hf/prefinetune"
    train_hf_dataset_name: str = "train.json"
    test_hf_dataset_name: str = "test.json"


# For Finetuning

@dataclass
class FineTuningDatasetConfig:
    df_path: str = 'Data/finetuning.csv'
    feature_col: str = 'complaint'
    label_col: str = 'issue'
    status_col: str = 'status'
    model_id: str = "tiiuae/falcon-7b"
    context_length: str = 512
    
    # === hf save directory ====
    hf_dir: str = "Data/hf/finetune"
    train_hf_dataset_name: str = "train.json"
    test_hf_dataset_name: str = "test.json"
    instruction_prompt = INSTRUCION_FINETUNING_PROMPT


# For Prefinetuning we just want to update the attention layer
# Or we can simply use the dense layer. Point being, we want to 
# feed the model different information

@dataclass
class PreFineTuningLoRAConfig:
    r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    bias: str = 'none'
    task_type: str = 'CAUSAL_LM'
    target_modules: List[str] = field(
        default_factory=lambda: ["query_key_value"]
    )


### FineTuning Config ###

@dataclass
class FineTuningLoRAConfig:
    r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.3
    bias: str = 'none'
    task_type: str = 'CAUSAL_LM'
    target_modules: List[str] = field(
        default_factory=lambda: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    )

# Add all the training arguments here
# For our case, we are trying to keep two different types of TrainingArguments 
# Because we might train our adapters for different cases. 

### For the prefinetuning step ###

@dataclass
class PreFineTuningTrainingConfig:
    output_dir: str = "./checkpoints/pre_finetuning_lora"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    optim: str = "paged_adamw_32bit"
    save_steps: int = 10
    save_total_limit: int = 3
    logging_steps: int = 10
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    max_steps: int = 15000
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "constant"
    report_to: str = "wandb"
    run_name: str = "prefinetuning-trial-2-15k"

### For the Finetuning step ###

@dataclass
class FineTuningTrainingConfig:
    output_dir: str = "./checkpoints/full"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    optim: str = "paged_adamw_32bit"
    save_steps: int = 10
    save_total_limit: int = 2
    logging_steps: int = 10
    learning_rate: float = 1e-5
    max_grad_norm: float = 0.3
    max_steps: int = 700
    warmup_ratio: float = 0.03
    report_to: str = "wandb"
    run_name: str = "full-finetuning"


@dataclass
class DuelAdapterInferenceConfig:
    base_model: str = 'tiiuae/falcon-7b'
    adapter_weights_paths: List[str] = field(
         default_factory=lambda: [
            'checkpoints/pre_finetuning_lora/checkpoint-240',
            'checkpoints/finetuning_lora/checkpoint-100',
        ]
    )
    max_new_tokens: int = 15
    temperature: float = 0.0
    top_k: int = 40
    top_p: int = 0.95

    # there are three options available
    # base: That will only evaluate the base model
    # prefinetune: This will only attach the prefinetune adapters
    # finetune: This will attach the finetuned adapters
    # all: This will attach all the adapters
    
    wandb_project_name = "Latest Project"
    experiment_type: str = "base"