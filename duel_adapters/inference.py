import re
import wandb
from tqdm.auto import tqdm  
from typing import List, Optional
from transformers import GenerationConfig

from duel_adapters.config import DuelAdapterInferenceConfig
from duel_adapters.merger import AdapterMerger
from duel_adapters.logger import logger


class Inference:
    def __init__(self, config: DuelAdapterInferenceConfig) -> None:
        self.config = config
        self.run = wandb.init(self.config.wandb_project_name)

        self.model, self.tokenizer = AdapterMerger.merge_adapters(
            base_model_id_or_path=self.config.base_model,
            adapter_paths=self.config.adapter_weights_paths
        )

        # Add the tokenizer configuration here only
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        logger.info(f'[INFERENCE]: Model {self.config.base_model} is loaded successfully with all the adapters')

    def regex_search(self, actual_text: str, generated_text: str) -> bool:
        pattern = re.compile(re.escape(actual_text), re.IGNORECASE)
        return True if pattern.search(generated_text) else False
    
    def quick_test(self, texts: List[str], labels: List[str]) -> None:
        """Does a quick test and provides the result in the form of texts
        Args:
            texts: List[str] A list of texts
            labels: List[str] A list of labels 
        """

        input_ids = self.tokenizer.batch_encode_plus(
            texts, return_tensors='pt', padding=True, truncation=True,
            max_length=512
        ).input_ids.to(self.model.device.type)
        
        model_outputs = self.tokenizer.batch_decode(
            self.model.generate(
                input_ids = input_ids,
                max_new_tokens = self.config.max_new_tokens, 
                pad_token_id = self.tokenizer.eos_token_id
            ), skip_special_tokens=True
        )

        for input, output, label in zip(texts, model_outputs, labels):
            start_index = output.find(input) + len(input)
            generated = output[start_index:]
            print("Generated text: ", generated)
            print("Actual text: ", label)
            print()
        logger.info("=> Finished testing")


    def log_inference(self, texts: List[str], labels: List[str], additional_prompt: Optional[str]=None):
        """Logs the prediction by the model to weights and bias
        Args:
            texts List[str]: A list of input strings that we want our model to do inference on
            labels List[str]: A list of label that we are expecting the model to generate
            additional_prompt str: Any additional system prompt that we can provide. 

        Note: We expect that when the experiment_type is not base then the texts should itself contain the instruction
        """
        additional_prompt = '' if additional_prompt is None else additional_prompt + '\n'
        contexts = [
            additional_prompt + text for text in texts
        ]

        # create the wandb Table
        table_name = f"Inference Table with {self.config.experiment_type} configuration"
        prediction_report = wandb.Table(column=[
            "Complaint",
            "Actual",
            "Predicted",
            "Label present in Predicted"
        ])

        for (text, label) in tqdm(zip(contexts, labels), total=len(texts)):
            
            input_ids = self.tokenizer(text, return_tensors='pt').input_ids.to(self.model.device.type)
            outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=GenerationConfig(
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p
                )
            )
            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction_report.add_data(
                text, 
                label,
                predicted_text,
                self.regex_search(label, predicted_text)
            )
        self.run.log({table_name: prediction_report})