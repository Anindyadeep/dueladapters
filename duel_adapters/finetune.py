import os
import warnings
from pathlib import Path 
from dataclasses import asdict 
from typing import Union, Tuple

import torch
import wandb
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from trl import SFTTrainer
from peft import LoraConfig

# module imports
from duel_adapters.config import (
    PreFineTuningLoRAConfig,
    FineTuningLoRAConfig,
    PreFineTuningTrainingConfig,
    FineTuningTrainingConfig,
    quantization_config
)

from duel_adapters import GeneralConfig

# warnings handeller
warnings.filterwarnings("ignore")

class FineTuner:
    def __init__(self, wandb_project_name: str) -> None: 
        self.run = wandb.init(project=wandb_project_name)
    
    def load_base_model(self, model_id: Union[str, Path]):
        # load the model and tokenizer and load the quant config from config 
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return model, tokenizer

    def load_configs(    
        self, 
        lora_config: Union[PreFineTuningLoRAConfig, FineTuningLoRAConfig],
        training_config: Union[PreFineTuningTrainingConfig, FineTuningTrainingConfig]
    ) -> Tuple[LoraConfig, TrainingArguments]:
        """
        Creates the Lora and the Training configs for finetuning prep using LoRA
        """
        loaded_lora_config = LoraConfig(**asdict(lora_config))
        loaded_training_config = TrainingArguments(**asdict(training_config))
        loaded_training_config.logging_dir = GeneralConfig().log_dir

        return loaded_lora_config, loaded_training_config
    
    def load_trl_trainer(
        self,
        model_id: str,
        lora_config: Union[PreFineTuningLoRAConfig, FineTuningLoRAConfig],
        training_config: Union[PreFineTuningTrainingConfig, FineTuningTrainingConfig],
        dataset_config: dict 
    ) -> Trainer:
        model, tokenizer = self.load_base_model(model_id)
        loaded_lora_config, loaded_training_config = self.load_configs(lora_config, training_config)
        trl_args = {
            'model': model,
            'tokenizer': tokenizer,
            'peft_config': loaded_lora_config,
            'args': loaded_training_config,
            **dataset_config
        }

        # do not use cache (kv-cache)
        model.config.use_cache = False
        trl_trainer = SFTTrainer(**trl_args)
        for name, module in trl_trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)
        return trl_trainer