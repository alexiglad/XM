import torch
from torch.utils.data import Dataset

class ImgSyntheticDataset(Dataset):
    def __init__(self, hparams, split="train"):
        self.hparams = hparams
        self.split = split
        self.use_cached_img_latents = hparams.use_cached_img_latents
        self.num_classes = hparams.num_classes
        
        if split == "train":
            self.length = 100000
        elif split in ["val", "valid", "validation"]:
            self.length = 10000
        else:
            self.length = 10000
        
        if self.use_cached_img_latents:
            assert hasattr(hparams, 'image_dims'), "image_dims required for cached latents"
            assert hparams.image_dims[0] % 8 == 0, "Image size must be divisible by 8 for VAE latents"
            latent_size = hparams.image_dims[0] // 8
            self.latent_shape = (4, latent_size, latent_size)
        else:
            self.image_shape = (3, hparams.image_dims[0], hparams.image_dims[1])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        torch.manual_seed(idx)
        
        label = torch.randint(0, self.num_classes, (1,)).item()
        
        if self.use_cached_img_latents:
            latent = torch.randn(self.latent_shape)
            return {'latent': latent, 'label': label}
        else:
            image = torch.randn(self.image_shape)
            return {'image': image, 'label': label}