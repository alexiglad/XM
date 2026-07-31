import torch
from torch.utils.data import Dataset


class VIDSyntheticDataset(Dataset):
    def __init__(self, hparams, size=10000000):
        self.hparams = hparams
        self.size = size
        self.context_length = hparams.context_length
        self.image_dims = hparams.image_dims
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if self.hparams.preencode_dataset:
            if self.hparams.backbone_type == "rae":
                # RAE with DINOv2-base: always encodes to 224x224, patch_size=14 -> 16x16 spatial, 768 channels
                input_size = 16
                in_channels = 768
            else:  # vae
                assert self.hparams.image_dims[0] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
                input_size = self.hparams.image_dims[0] // 8
                in_channels = 4
            random_embeddings = torch.randn(self.context_length, in_channels, input_size, input_size)
        else:
            random_embeddings = torch.randn(self.context_length, 3, self.image_dims[0], self.image_dims[1])
        return {"frames" : random_embeddings}