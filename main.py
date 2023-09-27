import wandb
from duel_adapters.finetune import FineTuner

# For prefinetuning
from duel_adapters import FalconDataset, PreFineTuneDatasetConfig
# configs for prefinetune
from duel_adapters.config import (
    PreFineTuneDatasetConfig,
    PreFineTuningLoRAConfig,
    PreFineTuningTrainingConfig
)

# configs for finetuning
from duel_adapters.config import (
    FineTuningDatasetConfig,
    FineTuningLoRAConfig,
    FineTuningTrainingConfig
)
from duel_adapters.finetune import FineTuner

# Inference
# from duel_adapters.config import DuelAdapterInferenceConfig
# from duel_adapters.inference import Inference

# logger
from duel_adapters import logger


from jsonargparse import CLI


# personal note:
# there can be another possibility of merged finetune, where we load the prefinetune lora weights as freezed weights 
# and finetune another layer on top of it
# this is not supported right now

def main(
    finetune_type: str = "prefinetune",
    wandb_project_name: str = 'ProjectDuelAdapterTrial1',
    model_id_or_path: str = 'tiiuae/falcon-7b',
    do_inference: bool = False) -> None:

    assert finetune_type in ['prefinetune', 'finetune', 'all'], ValueError("Available options: prefinetune, finetune, all")
    
    dataset = FalconDataset(
        config = PreFineTuneDatasetConfig() if finetune_type == 'prefinetune' else FineTuningDatasetConfig()
    ).export_hf_dataset(split='all')
    train_dataset, test_dataset = dataset['train'], dataset['test']

    # create the dataset config
    dataset_config = {
        'train_dataset': train_dataset,
        'eval_dataset': test_dataset,
        'dataset_text_field': 'text',
        'max_seq_length': 512
    }

    logger.info("[DATASET]: Dataset loaded successfully")

    # instantiate the lora and training config
    lora_config = PreFineTuningLoRAConfig() if finetune_type == 'prefinetune' else FineTuningLoRAConfig()
    training_config = PreFineTuningTrainingConfig() if finetune_type == 'prefinetune' else FineTuningTrainingConfig()

    # instantiate the finetuner object
    finetuner = FineTuner(wandb_project_name=wandb_project_name)
    trainer = finetuner.load_trl_trainer(
        model_id = model_id_or_path,
        lora_config = lora_config,
        training_config=training_config,
        dataset_config=dataset_config 
    )
    logger.info("[TRAINER] Training pipeline loaded successfully")
    trainer.train()
    wandb.finish()

    logger.info(f"[{finetune_type}] Process finish")

if __name__ == '__main__':
    CLI(main)