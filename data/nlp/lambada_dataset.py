from datasets import load_dataset
from torch.utils.data import Dataset
from functools import partial
from datasets import Dataset as hf_Dataset
import os
import json


class LambadaDataset(Dataset):
    def __init__(self, hparams, split): 
        self.hparams = hparams 
        hf_token = os.getenv('HF_TOKEN')
        hf_home = os.getenv('HF_HOME')
        dataset_dir = self.hparams.dataset_dir if self.hparams.dataset_dir != "" else hf_home
        self.dataset = load_dataset("EleutherAI/lambada_openai", "en", cache_dir=dataset_dir, token=hf_token, trust_remote_code=True)[split]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        words = text.split()
        
        context = " ".join(words[:-1])
        last_token = words[-1]
        
        if self.hparams.execution_mode == "inference":
            return context, last_token
        elif self.hparams.execution_mode == "pretrain":
            return text
        elif self.hparams.execution_mode == "finetune":
            return context, last_token
        else:
            raise ValueError(f"Execution mode not supported. Please add support for mode {self.hparams.execution_mode}")