from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from functools import partial
from datasets import Dataset as hf_Dataset
import os
import json

all_configs = [
    "elementary_math_qa",
    "simple_arithmetic_json",
    "dyck_languages",
    "matrixshapes",
    "repeat_copy_logic",
    "penguins_in_a_table",
]

class BigBenchDataset(Dataset):
    def __init__(self, hparams, split, task):
        self.hparams = hparams 

        hf_token = os.getenv('HF_TOKEN')
        hf_home = os.getenv('HF_HOME')
        dataset_dir = self.hparams.dataset_dir if self.hparams.dataset_dir != "" else hf_home
        
        if task == "all":
            self.dataset = load_dataset("tasksource/bigbench", all_configs[0], split=split, cache_dir=dataset_dir, token=hf_token, trust_remote_code=True)
            for subtask in all_configs[1:]:
                dataset = load_dataset("tasksource/bigbench", subtask, split=split, cache_dir=dataset_dir, token=hf_token, trust_remote_code=True)
                self.dataset = concatenate_datasets([self.dataset, dataset]) 
        else:
            if task not in all_configs:
                raise ValueError(f"Dataset not supported. Now supporting: {all_configs}")
            self.dataset = load_dataset("tasksource/bigbench", task, split=split, cache_dir=dataset_dir, token=hf_token, trust_remote_code=True)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.hparams.execution_mode == "inference":
            return f"[[Question]]: {self.dataset[idx]['inputs'].replace(' Answer:', '')}\n[[Answer]]: ", self.dataset[idx]['targets'][0]
        elif self.hparams.execution_mode == "pretrain":
            '''avoid duplicated "Answer: " '''
            return f"Question: {self.dataset[idx]['inputs'].replace(' Answer:', '')}\nAnswer: {self.dataset[idx]['targets'][0]}"
        elif self.hparams.execution_mode == "finetune":
            return f"[[Question]]: {self.dataset[idx]['inputs'].replace(' Answer:', '')}\n[[Answer]]: {self.dataset[idx]['targets'][0]}"
        else:
            raise ValueError(f"Execution mode not supported. Please add support for mode {self.hparams.execution_mode}")