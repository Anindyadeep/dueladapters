from pathlib import Path
from typing import Union, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from duel_adapters.config import quantization_config
from duel_adapters.logger import logger

           
class AdapterMerger:
    @classmethod
    def merge_adapters(cls, base_model_id_or_path: Union[str, Path], adapter_paths: List[str]):
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=base_model_id_or_path,
            return_dict=True,
            torch_dtype=torch.float16,
            trust_remote_code = True,
            device_map='auto',
            quantization_config=quantization_config # merge and unload is not working when using LoRA adapters
        )

        logger.info("=> Instantiated the base model")

        for adapter_path in adapter_paths:
            model = PeftModel.from_pretrained(base_model, adapter_path)
            logger.info(f"=> Peft model from path: {adapter_path} loaded successfully")
            # model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(base_model_id_or_path, trust_remote_code = True)
        return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = AdapterMerger.merge_adapters(
        base_model_id_or_path="tiiuae/falcon-7b",
        adapter_paths=[
            "/home/ec2-user/anindya/Experiments/Adapters/checkpoints/pre_finetuning_lora/checkpoint-240",
            "/home/ec2-user/anindya/Experiments/Adapters/checkpoints/finetuning_lora/checkpoint-100"
        ]
    )

    print(type(model))