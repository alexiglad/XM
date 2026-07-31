# Vendored from the official RAE implementation: https://github.com/bytetriper/RAE
# MIT License, Copyright (c) 2025 Boyang Zheng. See utils/rae/ATTRIBUTION.md.
from torch import nn
import torch
from math import *
from . import register_encoder
from transformers import SiglipModel

@register_encoder()
class SigLIP2wNorm(nn.Module):
    def __init__(self, model_name:str, num_tokens=256):
        super().__init__()
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.model = SiglipModel.from_pretrained(self.model_name).vision_model
        # remove the affine of final layernorm
        self.model.post_layernorm.elementwise_affine = False
        # remove the param
        self.model.post_layernorm.weight = None
        self.model.post_layernorm.bias = None
        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
    @torch.no_grad() # encoder is always frozen
    def forward(self, images):
        """
        images is of shape (B, C, H, W)
        where B is batch size, C is number of channels, H and W are height and
        """
        outputs = self.model(images, output_hidden_states=True, interpolate_pos_encoding = True)
        image_features = outputs.last_hidden_state
        return image_features