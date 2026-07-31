from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from functools import partial
from datasets import Dataset as hf_Dataset
import os
import json

all_configs = [
    "ARC-Easy",
    "ARC-Challenge",
]


class AI2ArcDataset(Dataset):
    def __init__(self, hparams, split, task="ARC-Challenge"):
        self.hparams = hparams 
        hf_token = os.getenv('HF_TOKEN')
        hf_home = os.getenv('HF_HOME')
        dataset_dir = self.hparams.dataset_dir if self.hparams.dataset_dir != "" else hf_home
        
        if task == "all":
            self.dataset = load_dataset("allenai/ai2_arc", all_configs[0], split=split, cache_dir=dataset_dir, token=hf_token, trust_remote_code=True)
            for subtask in all_configs[1:]:
                dataset = load_dataset("allenai/ai2_arc", subtask, split=split, cache_dir=dataset_dir, token=hf_token, trust_remote_code=True)
                self.dataset = concatenate_datasets([self.dataset, dataset]) 
        else:
            if task not in all_configs:
                raise ValueError(f"Dataset not supported. Now supporting: {all_configs}")
            self.dataset = load_dataset("allenai/ai2_arc", task, split=split, cache_dir=dataset_dir, token=hf_token, trust_remote_code=True)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        Q = self.dataset[idx]["question"] + "\n"
        for label, text in zip(self.dataset[idx]["choices"]["label"], self.dataset[idx]["choices"]["text"]):
            Q += f"{label}: {text}\n"
        A = self.dataset[idx]['answerKey']
        
        if self.hparams.execution_mode == "inference":
            return f"[[Question]]: {Q}\n[[Answer]]: ", A
        elif self.hparams.execution_mode == "pretrain":
            return f"Question: {Q}\nAnswer: {A}"
        elif self.hparams.execution_mode == "finetune":
            return f"[[Question]]: {Q}\n[[Answer]]: {A}"
        else:
            raise ValueError(f"Execution mode not supported. Please add support for mode {self.hparams.execution_mode}")
