import torch
from torch.utils.data import Dataset

class NLPSyntheticDataset(Dataset):
    def __init__(self, hparams, size=10000000, vocab_size=50257):
        self.size = size
        self.context_length = hparams.context_length
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        random_sequence = torch.randint(0, self.vocab_size, (self.context_length,))
        return {"input_ids": random_sequence}