import os 
from typing import Optional, Union

import pandas as pd 
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from duel_adapters.config import PreFineTuneDatasetConfig, FineTuningDatasetConfig

# Stored different adapters for different things

class FalconDataset:
    def __init__(self, config: Union[PreFineTuneDatasetConfig, FineTuningDatasetConfig]) -> None:
        self.config = config
        self.data = pd.read_csv(self.config.df_path, index_col=0)
        self.train_data, self.test_data = self.data[self.data[self.config.status_col] == 'train'], self.data[self.data[self.config.status_col] == 'test']
        self.using_instruction_tuning = False if isinstance(config, PreFineTuneDatasetConfig) else True

    def get_hf_dataset(self, data: pd.DataFrame, split: Optional[str] = 'train', save: Optional[bool]=True) -> Dataset:
        """
        Get the Huggingface formatted dataset
        Args:
            data: (pd.DataFrame) The dataframe of choice
            split: (Optional[str]) The split of the dataset
            save: (Optional[bool]) Whether to save or not
        
        If `save` is set to True, then it will save according to the config file. 
        """
        features = data[self.config.feature_col].tolist()
        labels = data[self.config.label_col].tolist()

        if not self.using_instruction_tuning:
            merged = [
                feature + ('' if split == 'test' else label) 
                for (feature, label) in zip(features, labels)
            ]
        else:
            merged = [
                self.config.instruction_prompt.format(
                    text=feature, label=label if split == "train" else ""
                ) for (feature, label) in zip(features, labels)
            ]

        if split == 'train':
            dataset = Dataset.from_dict({'text': merged}, split='train')
        else:
            dataset = Dataset.from_dict({
                'text': merged,
                'label': labels
            }, split='train')


        if save:
            if not os.path.isdir(self.config.hf_dir):
                os.makedirs(self.config.hf_dir, exist_ok=True)
            
            if not os.path.exists(os.path.join(self.config.hf_dir, f'{split}.json')):
                dataset.save_to_disk(os.path.join(self.config.hf_dir, f'{split}.json'))
        return dataset 
        
    def export_hf_dataset(self, split: str) -> Union[Dataset, DatasetDict]:
        """
        Exports the data to huggingface format.
        Args:
            split (str): Supports two values (train/test/all), accordingly the dataset will be exported. If 'all'
            is choosen then it will return a DatasetDict containing both the split
        Returns:
            Union[Dataset, DatasetDict]: Returns Dataset object when using split = train/text else DatasetDict
        """
        assert split in ['train', 'test', 'all'], ValueError("Supported keys: ['train', 'test', 'all']")
        if split in ['train', 'test']:
            return self.get_hf_dataset(
                self.train_data if split == 'train' else self.test_data,
                split = split
            )
        else:
            return DatasetDict({
                'train': self.get_hf_dataset(self.train_data, split='train'),
                'test': self.get_hf_dataset(self.test_data, split='test')
            })   