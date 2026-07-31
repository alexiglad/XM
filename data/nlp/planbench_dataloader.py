from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from functools import partial
from datasets import Dataset as hf_Dataset
import os
import json

all_configs = [
    'task_1_plan_generation',
    'task_2_plan_optimality',
    'task_3_plan_verification',
    'task_5_plan_generalization',
    'task_7_plan_execution',
    'task_8_1_goal_shuffling',
    'task_8_2_full_to_partial'   
]

class PlanBenchDataset(Dataset):
    def __init__(self, hparams, split='train', task='task_1_plan_generation'):
        self.hparams = hparams
        hf_token = os.getenv('HF_TOKEN')
        hf_home = os.getenv('HF_HOME')
        dataset_dir = self.hparams.dataset_dir if self.hparams.dataset_dir != "" else hf_home
        
        
        if task == "all":
            '''selecting all doesn't work for now'''
            raise ValueError("selecting all isn't supported for now")
            self.dataset = load_dataset("tasksource/planbench", all_configs[0], split=split, cache_dir=dataset_dir, token=hf_token, trust_remote_code=True)
            for subtask in all_configs[1:]:
                dataset = load_dataset("tasksource/planbench", subtask, split=split, cache_dir=dataset_dir, token=hf_token, trust_remote_code=True)
                self.dataset = concatenate_datasets([self.dataset, dataset]) 
        else:
            if task not in all_configs:
                raise ValueError(f"Dataset not supported. Now supporting: {all_configs}")
            self.dataset = load_dataset("tasksource/planbench", task, split=split, cache_dir=dataset_dir, token=hf_token,trust_remote_code=True)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        if self.hparams.execution_mode == "inference":
            return f"[[Question]]: {self.dataset[idx]['query']}\n[[Answer]]: ", self.dataset[idx]['ground_truth_plan'][0]  
        elif self.hparams.execution_mode == "pretrain":
            return f"Question: {self.dataset[idx]['query']} Answer: {self.dataset[idx]['ground_truth_plan'][0]}"
        elif self.hparams.execution_mode == "finetune":
            return f"[[Question]]: {self.dataset[idx]['query']} [[Answer]]: {self.dataset[idx]['ground_truth_plan'][0]}"
        else:
            raise ValueError(f"Execution mode not supported. Please add support for mode {self.hparams.execution_mode}")