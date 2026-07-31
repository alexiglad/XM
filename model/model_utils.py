import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy
from torchvision.transforms import functional as TF
import torchvision.models as models
from diffusers import AutoencoderKL
import math
import random
import numpy as np
from functools import partial
from PIL import Image
import torchvision
from torchvision.transforms import ToPILImage
from contextlib import contextmanager
from datetime import datetime, timedelta
import torch.distributed as dist
from torch.distributed.distributed_c10d import _set_pg_timeout
from typing import Optional, Tuple
from dataclasses import dataclass
import os
from torchvision.utils import make_grid, save_image
import json
from pytorch_lightning.utilities.rank_zero import rank_zero_only


model_sizes = { # small -> xl same as mamba https://arxiv.org/pdf/2312.00752; all others estimated empirically. LRs based off mamba where applicable
    "4xs": { # LR 0.0024 recommended
        "num_transformer_blocks": 2,
        "multiheaded_attention_heads": 2,
        "embedding_dim": 128,
    },
    "3xs": { # LR 0.0018
        "num_transformer_blocks": 4,
        "multiheaded_attention_heads": 4,
        "embedding_dim": 256,
    },
    "xxs": { # LR 0.0012
        "num_transformer_blocks": 6,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "2xs": { # LR 0.0012
        "num_transformer_blocks": 6,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "xs": { # LR 0.0009
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "small": { # LR 0.0006
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 12,
        "embedding_dim": 768,
    },
    "medium": { # 0.0003
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1024,
    },
    "large": { # 0.00025
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1536,
    },
    "xl": { # 0.0002
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 32,
        "embedding_dim": 2048,
    },
    "vit_small": {  # LR 0.0001
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "vit_base": {  # 0.0001
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 12,
        "embedding_dim": 768,
    },
    "vit_large": {  # 0.0001
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1024,
    },
    "vit_xl": {  # 0.0001
        "num_transformer_blocks": 28,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1152,
    },
    "vit_2xl": {  # 0.0001
        "num_transformer_blocks": 32,
        "multiheaded_attention_heads": 24,
        "embedding_dim": 1536,
    },
}




class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        out = self.dropout(out)
        return x + out  # Add the residual connection

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, final_size, dropout_rate, layer_norm, num_hidden_layers=1):
        super(MLP, self).__init__()
        self.add_residual_connections = True  # Residual connections are always on by default
        self.layers = nn.ModuleList()

        # Initial layer
        self.layers.append(nn.Linear(input_size, hidden_size, bias=False))
        if layer_norm:
            self.layers.append(nn.LayerNorm(hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(1, num_hidden_layers - 1):
            add_residual = self.add_residual_connections and i % 2 == 0

            if add_residual:
                self.layers.append(ResidualBlock(hidden_size, dropout_rate))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
                self.layers.append(nn.ReLU())

            self.layers.append(nn.Dropout(dropout_rate))

        # Last layer
        if final_size == hidden_size and self.add_residual_connections and (num_hidden_layers - 1) % 2 == 0:
            self.layers.append(ResidualBlock(hidden_size, dropout_rate))
        else:
            self.layers.append(nn.Linear(hidden_size, final_size, bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def format_lr(lr: float) -> str:
    s = f"{lr:.1e}"
    m, e = s.split("e")
    e = str(int(e))
    if m.endswith(".0"):
        m = m[:-2]
    return f"{m}e{e}"


def _collapse_dupes(seq):
    if not seq:
        return seq
    first = seq[0]
    for x in seq[1:]:
        if x != first:
            return seq
    return [first]


def _format_var_value(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, (list, tuple)):
        vals = [str(x) for x in _collapse_dupes(list(v))]
        if len(vals) == 1:
            return vals[0]
        return "x".join(vals)
    if isinstance(v, str):
        toks = v.split()
        if len(toks) > 1:
            toks = [str(x) for x in _collapse_dupes(toks)]
            if len(toks) == 1:
                return toks[0]
            return "x".join(toks)
        return v
    return str(v)


def format_run_name_hparams(hparams, args):
    """Returns a flat list of formatted hparam strings (e.g. ['img=256', 'ps=2', 'cfg=0.0x1.5'])."""
    ctx = dict(os.environ)
    for k, v in vars(args).items():
        ctx[k] = _format_var_value(v)

    items = []
    for h in hparams:
        for part in h.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                key = k.strip()
                val = v.strip()
                if val in ctx:
                    val = ctx[val]
                items.append(f"{key}={val}")
            else:
                items.append(part)
    return items


DEFAULT_DDP_TIMEOUT = timedelta(minutes=30)


@contextmanager
def temporary_ddp_timeout(timeout, restore=DEFAULT_DDP_TIMEOUT):
    """Bump the default process group's collective timeout for a slow section (preencoding, online eval), then restore. No-op when dist isn't initialized (single-GPU)."""
    if not dist.is_initialized():
        yield
        return
    _set_pg_timeout(timeout)
    try:
        yield
    finally:
        _set_pg_timeout(restore)


@rank_zero_only # to ensure only one wandb run is created, if didnt do that then each GPU would create its own wandb run
def setup_wandb(args, resume_run_id=None):
    import wandb
    if wandb.run is None:
        init_kwargs = dict(
            dir="logs/",
            name=f"{args.wandb_run_name}",
            entity=f"{args.wandb_entity}",
            project=f"{args.wandb_project}",
            mode="offline" if args.wandb_offline else "online",
        )
        if resume_run_id is not None: # "allow" (vs "must") won't hard-fail if the run is missing
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "allow"
            print(f"resuming wandb run {resume_run_id}")
        run = wandb.init(**init_kwargs)
        wandb.define_metric("__init", hidden=True) # this is used to force wandb to start tracking stdout in logs
        wandb.define_metric("*", step_metric="trainer/global_step") # default x axis is trainer/global_step
        return run
    return None


def load_wandb_run_id(ckpt_path): # pulls wandb_run_id from PL ckpt so resumed training can continue the same wandb run; returns None on any failure so caller falls back to a fresh run
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt.get("wandb_run_id", None)
    except Exception as e:
        print(f"WARNING: failed to read wandb_run_id from {ckpt_path} ({e}); will start a new wandb run")
        return None


def log_pred_futures(futures, device, dataset_name, i, denormalize):
    denormalized_futures = denormalize(futures.clone(), dataset_name, device = device)

    to_pil = ToPILImage()
    for b in range(denormalized_futures.shape[0]):  # Loop over the batch size
        if b % 16 == 0:
            for s in range(denormalized_futures.shape[1]):  # Loop over the sequence length
                frame_to_save = to_pil(denormalized_futures[b, s].cpu())  # Extract a frame (C x W x H)
                
                # Save the image
                current_time = datetime.now().strftime("%H_%M_%S")
                frame_to_save.save(f"./logs/debug/mcmc_futures/{current_time}_batch_{b}_seq_{s}_dev_{device}_iter_{i}.png")

def denormalize(tensor, dataset_name, device, custom_normalization, vae_normalization=False):
    tensor = tensor.clone().detach()

    # Define default normalization values
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    default_mean = torch.tensor(default_mean, device=device).view(1, 1, 3, 1, 1)
    default_std = torch.tensor(default_std, device=device).view(1, 1, 3, 1, 1)
    # Dataset-specific normalization lookup
    if custom_normalization:
        normal_lookup = {
            "k400": ([1.00370078, 0.99871626, 0.97407404], [-0.24295556, -0.24931058, -0.13959686]),
            "smth": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
            "ImageNet": ([1, 1, 1], [0, 0, 0]),
            "something": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
            "ImageNet1k": ([1, 1, 1], [0, 0, 0])
        }
        dataset_std, dataset_mean = normal_lookup.get(dataset_name, ([1, 1, 1], [0, 0, 0]))

        # Convert means and stds to tensors and reshape for broadcast compatibility
        dataset_mean = torch.tensor(dataset_mean, device=device).view(1, 1, 3, 1, 1)
        dataset_std = torch.tensor(dataset_std, device=device).view(1, 1, 3, 1, 1)
        

        # Perform denormalization
        # First reverse the dataset-specific normalization
        tensor = tensor * dataset_std + dataset_mean
    
    # Then reverse the default normalization
    if vae_normalization:
        default_mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 1, 3, 1, 1) 
        default_std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 1, 3, 1, 1)
        tensor = tensor * default_std + default_mean
        return tensor.clamp(0, 1) # without this clamp images often have visual artifacts/have high intensity pixels randomly
    else:
        return (tensor * default_std + default_mean).clamp(0, 1)

def scale_clamp(tensor, min_value, max_value):
    scale_factor = torch.ones_like(tensor)
    scale_factor = torch.where(tensor > max_value, tensor / max_value, scale_factor)
    scale_factor = torch.where(tensor < min_value, tensor / min_value, scale_factor)
    
    scaled_tensor = tensor / scale_factor
    return scaled_tensor

def load_trained_pl_model(ckpt_path, new_hparams = None, for_inference = False, return_pretrained_hparams=False, override_pretrained_params = {}):
    from base_model_trainer import ModelTrainer
    checkpoint = torch.load(ckpt_path, weights_only=False)
    pretrained_hparams = checkpoint['hyper_parameters']
    
    if new_hparams is not None:
        model = ModelTrainer(new_hparams)
    else:
        for k, v in override_pretrained_params.items():
            pretrained_hparams[k] = v
        model = ModelTrainer(pretrained_hparams)
    model.load_state_dict(checkpoint['state_dict'])

    if for_inference:
        model.cuda().eval()
        model.model.eval()
    if not return_pretrained_hparams:
        return model.model
    else:
        return model.model, pretrained_hparams

def print_model_layers_and_status(model):
    for name, module in model.named_modules():
        print(f'Layer: {name}, Type: {type(module).__name__}, Training Mode: {module.training}')

def init_weights(model, weight_initialization_method, nonlinearity='linear', weight_initialization_gain=1.0):
    def _init_weights(m):
        if isinstance(m, nn.Embedding):
            if weight_initialization_method == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
            elif weight_initialization_method == "xavier":
                nn.init.xavier_normal_(m.weight)
            else:
                raise ValueError(f"Unknown weight init method: {weight_initialization_method}")
            if weight_initialization_gain != 1.0:
                m.weight.data *= weight_initialization_gain
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, nn.Linear):
            if weight_initialization_method == "he":
                valid_nonlinearities = ['linear', 'relu', 'leaky_relu', 'selu', 'tanh']
                if nonlinearity not in valid_nonlinearities:
                    raise ValueError(f"Unsupported nonlinearity: {nonlinearity}. Must be one of {valid_nonlinearities}")

                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif weight_initialization_method == "xavier":
                nn.init.xavier_normal_(m.weight)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            else:
                raise ValueError(f"Unknown weight init method: {weight_initialization_method}")
    
    model.apply(_init_weights)


def load_image_encoder(backbone_type, backbone_size, device=None, use_ema=False):
    # use_ema only applies to sdxl vae

    vit_backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
        
    if backbone_type == 'dinov2':
        backbone_name = vit_backbone_archs[backbone_size]
        model = torch.hub.load('facebookresearch/dinov2', model=f"dinov2_{backbone_name}")
        del model._parameters['mask_token'] # this is done as this param was unused and was causing pl ddp unused param issues
    elif backbone_type == "vae": # all have same encoder just different decoder
        if use_ema:
            model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
        else:
            model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    elif backbone_type == "rae":
        from utils.rae.stage1 import RAE

        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_dir, 'models', 'rae')
        decoder_weights = os.path.join(models_dir, 'decoders', 'dinov2', 'wReg_base', 'ViTXL_n05', 'dinov2_decoder.pt')
        stat_path = os.path.join(models_dir, 'stats', 'dinov2', 'wReg_base', 'imagenet1k', 'stat.pt')
        decoder_config_dir = os.path.join(models_dir, 'configs', 'decoder', 'ViTXL')

        # Download weights from HuggingFace if not present
        if not os.path.exists(decoder_weights) or not os.path.exists(stat_path):
            print("Downloading RAE model weights from HuggingFace...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id='nyu-visionx/RAE-collections',
                local_dir=models_dir,
                allow_patterns=[
                    'decoders/dinov2/wReg_base/ViTXL_n05/*',
                    'stats/dinov2/wReg_base/imagenet1k/*',
                ]
            )

        # Write decoder config.json if missing (needed by AutoConfig.from_pretrained)
        config_json_path = os.path.join(decoder_config_dir, 'config.json')
        if not os.path.exists(config_json_path):
            os.makedirs(decoder_config_dir, exist_ok=True)
            vitxl_config = {
                "architectures": ["ViTMAEForPreTraining"],
                "attention_probs_dropout_prob": 0.0,
                "decoder_hidden_size": 1152, "decoder_intermediate_size": 4096,
                "decoder_num_attention_heads": 16, "decoder_num_hidden_layers": 28,
                "hidden_act": "gelu", "hidden_dropout_prob": 0.0,
                "hidden_size": 768, "image_size": 224,
                "initializer_range": 0.02, "intermediate_size": 3072,
                "layer_norm_eps": 1e-12, "mask_ratio": 0.75,
                "model_type": "vit_mae", "norm_pix_loss": False,
                "num_attention_heads": 12, "num_channels": 3,
                "num_hidden_layers": 12, "patch_size": 16,
                "torch_dtype": "float32", "transformers_version": "4.42.3"
            }
            with open(config_json_path, 'w') as f:
                json.dump(vitxl_config, f, indent=2)

        model = RAE(
            encoder_cls='Dinov2withNorm',
            encoder_config_path='facebook/dinov2-with-registers-base',
            encoder_input_size=224,
            encoder_params={'dinov2_path': 'facebook/dinov2-with-registers-base', 'normalize': True},
            decoder_config_path=decoder_config_dir,
            pretrained_decoder_path=decoder_weights,
            noise_tau=0.,  # no noise at inference
            reshape_to_2d=True,
            normalization_stat_path=stat_path,
        )
    else:
        raise NotImplementedError(f"Unspported backbone type: {backbone_type}")
    if device is not None:
        model = model.to(device)
    return model
    
def get_encoded_images(batch, backbone_type, image_encoder, sdxl_vae_standardization = True):
    with torch.no_grad():
        if backbone_type == 'dinov2':
            return image_encoder(batch)
        elif backbone_type == "vae":
            if not sdxl_vae_standardization:
                return image_encoder.encode(batch).latent_dist.mean
            elif sdxl_vae_standardization:
                return image_encoder.encode(batch).latent_dist.mean.mul_(0.18215) # constant following SDXL VAE https://github.com/CompVis/stable-diffusion
        elif backbone_type == "rae":
            return image_encoder.encode(batch)  # RAE normalizes internally
        else:
            raise NotImplementedError(f"have not yet implemented backbone_type: {backbone_type}")

def reset_image_encoder_decoder(hparams, model): # for if image_encoder is a vae
    model.image_encoder = load_image_encoder(hparams.backbone_type, hparams.vit_backbone_size, use_ema=True).to(model.device)
    model.image_encoder.eval()
    for param in model.image_encoder.parameters():
        param.requires_grad = False


def decode_embeddings(model, samples_dict):
    """Decode a dict of embedding tensors to pixel space using model.image_encoder.
    samples_dict values should be (C,H,W) for images or (S,C,H,W) for videos."""
    names = list(samples_dict.keys())
    embeddings = [samples_dict[n] for n in names]
    is_video = embeddings[0].ndim == 4

    if is_video:
        seq_length = embeddings[0].shape[0]
        embed_shape = embeddings[0].shape[1:]
        n_samples = len(embeddings)
        flat = torch.stack(embeddings, dim=0).reshape(-1, *embed_shape)
    else:
        n_samples = len(embeddings)
        flat = torch.stack(embeddings, dim=0)

    if model.hparams.backbone_type == "rae":
        decoded = model.image_encoder.decode(flat).clamp(0, 1)
    else:
        decoded = model.image_encoder.decode(flat / 0.18215).sample
        decoded = denormalize(decoded.unsqueeze(0), model.hparams.dataset_name, model.device, model.hparams.custom_image_normalization, True).squeeze(0)

    if is_video:
        image_shape = decoded.shape[1:]
        decoded = decoded.reshape(n_samples, seq_length, *image_shape)

    return {name: decoded[i] for i, name in enumerate(names)}


def log_generated_samples(model, phase, log_dict, generate_samples_fn, log_during_val=False):
    """Logging wrapper for image/video generation. Handles step checking, encoder reset, decoding, and val logging.
    generate_samples_fn: callable() -> dict of {name: embedding_tensor} to decode.
    log_during_val: if True, also logs during the first val step after a train log."""
    log_every = model.hparams.log_video_every_n_steps if model.hparams.modality == "VID" else model.hparams.log_image_every_n_steps
    is_update_step = (model.trainer.fit_loop.epoch_loop.batch_progress.current.completed + 1) % model.trainer.accumulate_grad_batches == 0
    should_log_train = model.trainer.global_step % log_every == 0 and is_update_step and phase == "train"
    assert not log_during_val or model.hparams.modality == "VID", "log_during_val is only supported for VID models"
    should_log_val = log_during_val and phase == "valid" and model.log_val_video

    if should_log_train:
        if log_during_val:
            model.log_val_video = True
        if not model.reset_image_encoder_decoder:
            reset_image_encoder_decoder(model.hparams, model)
            model.reset_image_encoder_decoder = True

    if should_log_train or should_log_val:
        if should_log_val:
            model.log_val_video = False
        with torch.no_grad():
            samples_dict = generate_samples_fn()
            log_dict.update(decode_embeddings(model, samples_dict))


def hinged_mse_loss(predictions, targets, margin=0.1):
    """
    Compute the Hinged MSE loss between predictions and targets.
    :param predictions: Predicted values.
    :param targets: Ground truth values.
    :param margin: The threshold below which errors are ignored.
    :return: Hinged MSE loss.
    """
    errors = torch.abs(predictions - targets)
    hinged_errors = torch.where(errors > margin, errors, torch.zeros_like(errors))
    loss = torch.mean(hinged_errors ** 2)
    return loss

def find_subsequences(input_tensor, sub_seq):
    sub_seq_len = len(sub_seq)
    batch_size, seq_len = input_tensor.shape
    sub_seq_tensor = torch.tensor(sub_seq, device=input_tensor.device)
    sub_seq_tensor = sub_seq_tensor.view(1, -1)
    windows = input_tensor.unfold(1, sub_seq_len, 1)
    matches = (windows == sub_seq_tensor).all(dim=2).long()
    
    if not matches.any(dim=1).all():
        raise ValueError("Sub-sequence not found in one or more sequences.")
    
    start_positions = matches.argmax(dim=1)
    return start_positions

def mask_q_tokens(input_tensor, tokenizer):
    '''
    input_tensor = [batch size, seq len]
    '''
    batch_size = input_tensor.shape[0]
    seq_length = input_tensor.shape[1]
    answer_tag = tokenizer.encode("[[Answer]]:", add_special_tokens=True)
    
    answer_start_pos = find_subsequences(input_tensor, answer_tag)
    answer_start_pos += len(answer_tag)
    mask = torch.arange(seq_length, device=input_tensor.device).expand(batch_size, seq_length)
    mask = mask < answer_start_pos.unsqueeze(1)
    input_tensor = torch.where(mask, tokenizer.pad_token_id, input_tensor)
    
    return input_tensor

def analyse_tokens(input_tensor, tokenizer):
    '''for debugging only'''
    decode = tokenizer.batch_decode(input_tensor, skip_special_tokens=True)
    for i in range(input_tensor.shape[0]):
        print(input_tensor[i].tolist())
        print(decode[i])
        print('-'*60)

def setup_ar_transformer(hparams): # specifically for baseline transformer
    from model.ar_transformer import Transformer, TransformerModelArgs

    max_seq_len = hparams.context_length if hparams.xm_best_of_k <= 1 else hparams.context_length + 1 # need + 1 for the extra best-of-k embed
    transformer_args = TransformerModelArgs(dim = hparams.embedding_dim, n_layers = hparams.num_transformer_blocks, n_heads = hparams.multiheaded_attention_heads, max_batch_size = hparams.batch_size_per_device, max_seq_len=max_seq_len, weight_initialization = hparams.weight_initialization_method, ffn_dim_multiplier=hparams.ffn_dim_multiplier, weight_initialization_gain=hparams.weight_initialization_gain)
    transformer = Transformer(params=transformer_args)
    return transformer

def setup_3d_transformer(hparams, is_energy=False, use_adaln=False): # specifically for baseline transformer
    from model.video_transformer_3d import VisionTransformer

    if hparams.backbone_type == "vae":
        in_chans = 4
        latent_spatial = (hparams.image_dims[0] // 8, hparams.image_dims[1] // 8)
    elif hparams.backbone_type == "rae":
        in_chans = 768  # RAE with DINOv2 produces 768-channel latents
        latent_spatial = (16, 16)  # RAE always produces 16x16 spatial grid
    else:
        raise ValueError(f"setup_3d_transformer does not support backbone_type={hparams.backbone_type}")

    transformer = VisionTransformer(img_size=latent_spatial, patch_size=hparams.patch_size, num_frames=hparams.context_length, tubelet_size=1, in_chans=in_chans, embed_dim=hparams.embedding_dim, depth=hparams.num_transformer_blocks, num_heads=hparams.multiheaded_attention_heads, mlp_ratio=hparams.ffn_dim_multiplier, use_rope=True, unembed_patches=True, is_energy=is_energy, use_adaln=use_adaln)
    return transformer

def setup_bi_transformer(hparams): # specifically for baseline transformer
    from model.bi_transformer import Transformer, TransformerModelArgs
    all_indices = range(hparams.context_length)
    non_cross_attention_indices = [i for i in all_indices if i not in hparams.world_modeling_given_frames]
    
    transformer_args = TransformerModelArgs(dim = hparams.embedding_dim, n_layers = hparams.num_transformer_blocks, n_heads = hparams.multiheaded_attention_heads, max_batch_size = hparams.batch_size_per_device, max_seq_len=hparams.context_length, weight_initialization = hparams.weight_initialization_method, ffn_dim_multiplier=hparams.ffn_dim_multiplier, weight_initialization_gain=hparams.weight_initialization_gain, cross_attention_transformer=hparams.cross_attention_transformer, cross_attention_condition_indices=hparams.world_modeling_given_frames, non_cross_attention_indices=non_cross_attention_indices)
    transformer = Transformer(params=transformer_args)
    return transformer

def has_layer_norm(model):
    return any(isinstance(module, nn.LayerNorm) for _, module in model.named_modules())

def init_wandb_watch(wandb_logger, model_trainer, wandb_watch_log_freq):
    if not has_layer_norm(model_trainer.model):
        wandb_logger.watch(model_trainer.model, log="all", log_freq = wandb_watch_log_freq)
    
    else: # all of complex below code is to get around the issue where wandb watch with layer norm has 'AttributeError: 'NoneType' object has no attribute 'data'' when logging gradients...
        non_layernorm_container = nn.Module()
        layernorm_container = nn.Module()

        non_ln_modules = {}
        ln_modules = {}

        for name, module in model_trainer.model.named_modules():
            if name == "": # skips top level model
                continue
            safe_name = name.replace(".", "_") # model cant contain '.' in name

            if isinstance(module, nn.LayerNorm):
                ln_modules[safe_name] = module
            else:
                # Only add modules that don't contain LayerNorm as submodules
                has_ln_child = any(isinstance(child, nn.LayerNorm) 
                                for child in module.modules())
                if not has_ln_child:
                    non_ln_modules[safe_name] = module

        for name, module in non_ln_modules.items():
            non_layernorm_container.add_module(name, module)

        for name, module in ln_modules.items():
            layernorm_container.add_module(name, module)

        # print("\nNon-LayerNorm modules:")
        # for name, _ in non_layernorm_container.named_modules():
        #     if name != "":  # Skip the container itself
        #         print(f"  - {name}")

        # print("\nLayerNorm modules:")
        # for name, _ in layernorm_container.named_modules():
        #     if name != "":  # Skip the container itself
        #         print(f"  - {name}")

        wandb_logger.watch(non_layernorm_container, log="all", log_freq=wandb_watch_log_freq)
        wandb_logger.watch(layernorm_container, log="parameters", log_freq=wandb_watch_log_freq)

def save_frames(tensor, root_dir, subfolder, start_index=0):
    to_pil = ToPILImage()
    subfolder_path = os.path.join(root_dir, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)

    all_frames = [] if subfolder == 'debug' else None

    for i, video in enumerate(tensor):
        if subfolder != 'debug':
            video_dir = os.path.join(subfolder_path, f'video_{start_index + i}')
            os.makedirs(video_dir, exist_ok=True)

        for j, frame in enumerate(video):
            frame_img = to_pil(frame.cpu().detach())

            if subfolder != 'debug':
                frame_img.save(os.path.join(video_dir, f'frame_{j}.png'))
            else:
                all_frames.append(frame_img)

    if subfolder == 'debug':
        frame_tensors = torch.stack([torchvision.transforms.functional.to_tensor(img) for img in all_frames])
        grid = make_grid(frame_tensors, nrow=int(len(frame_tensors)**0.5))
        save_image(grid, os.path.join(subfolder_path, f'{start_index}.png'))

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def init_dit_final_layer(dit, std):
    """Re-initialize the DiT final layer with specified std for high-variance outputs."""
    nn.init.normal_(dit.final_layer.linear.weight, std=std)
    nn.init.normal_(dit.final_layer.linear.bias, std=std)

def setup_diffusion_transformer(hparams):
    from model.diffusion_transformer import DiT
    assert hparams.image_dims[0] == hparams.image_dims[1], "need to use square image with current implementation"
    
    if hparams.backbone_type == "rae":
        # RAE with DINOv2-base: always encodes to 224x224, patch_size=14 -> 16x16 spatial, 768 channels
        input_size = 16
        in_channels = 768
    else:  # vae
        assert hparams.image_dims[0] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        input_size = hparams.image_dims[0] // 8
        in_channels = 4

    dit = DiT(input_size=input_size, patch_size=hparams.patch_size, in_channels=in_channels, hidden_size=hparams.embedding_dim, depth=hparams.num_transformer_blocks, num_heads=hparams.multiheaded_attention_heads, mlp_ratio=hparams.ffn_dim_multiplier, learn_sigma=hparams.learn_sigma)
    return dit

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# note this is unused for text conditional instead of class conditional https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, is_training, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (is_training and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



# all pos enc functions from below are from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py, from DiT codebase
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def update_ema_model(students, teachers, ema_momentum): # gotten from dino codebases https://github.com/facebookresearch/dino
	assert len(students) == len(teachers), "there should be a teacher for every student!"
	with torch.no_grad():
		for student, teacher in zip(students, teachers):
			if student is None and teacher is None:
				continue
			if (student is None) != (teacher is None):
				raise ValueError("Student and teacher types differ")
            
			if isinstance(student, nn.Parameter):
				assert isinstance(teacher, nn.Parameter), "EMA pair types must match"
				teacher.data.mul_(ema_momentum)
				teacher.data.add_((1.0 - ema_momentum) * student.data)
				continue
			student_params = dict(student.named_parameters())
			teacher_params = dict(teacher.named_parameters())
			assert student_params.keys() == teacher_params.keys(), "Student/teacher param mismatch"
			for name in teacher_params.keys():
				# teacher = m*teacher + (1-m)*student
				teacher_params[name].data.mul_(ema_momentum)
				teacher_params[name].data.add_((1.0 - ema_momentum) * student_params[name].data)

def check_model_ema_params(model): # check to make sure all learnable params are being ema'ed properly
    for m in model.ema_model_submodules:
        if isinstance(m, nn.Module):
            for p in m.parameters():
                p.requires_grad = False
        elif isinstance(m, nn.Parameter):
            m.requires_grad = False

    learnable_children = sum(1 for n, m in model.named_children() if not n.startswith('ema_') and any(p.requires_grad for p in m.parameters()))
    top_level_params = sum(1 for n, p in model.named_parameters(recurse=False) if p.requires_grad and not n.startswith('ema_'))
    assert len(model.regular_model_submodules) == len(model.ema_model_submodules) == (learnable_children + top_level_params), "EMA submodule lists must cover all learnable modules and parameters"


def xm_chunked_best_of_k(model_forward, loss_calc_wrapper, conditions, gt_samples, best_of_k, max_chunk_bs_mult, save_mem_mode = True, debug_save_mem_mode = False, not_training = False, **loss_calc_kwargs):
    """
    Parallelizes Explorative Modeling over chunks to find the best-of-k loss mode.

    This function explores multiple potential solutions in parallel (chunks) and selects the one with the minimum loss.
    Chunks allow simulating multiple batches at once for parallelization. Usually just returns losses but can also return predictions

    NOTE determinism contract: loss_calc_wrapper must be a deterministic function of
    (conditions, gt_samples, rand_inputs, rand_seeds, **loss_calc_kwargs) -- all stochasticity
    (cfg dropout masks, jump indices, per-position randomness, etc) must be derived from these
    arguments and passed in, never sampled inside the wrapper from the global RNG. save_mem_mode
    relies on this: it re-runs the wrapper with the winning rand_inputs/rand_seeds to rebuild the
    autograd graph, so any unseeded randomness means the recomputed loss silently belongs to a
    different candidate than the one selected (xm_debug_mode catches this).

    NOTE CFG/dropped-label nuance: parallelized exploration + save_mem_mode complicates CFG dropout a tad.
    The drop decision must be (1) sampled once per original sample (shape regular_bs) outside the wrapper
    and passed in via kwargs (e.g. cfg_drop_mask), then tiled across the K candidates -- so all candidates
    for a sample share the same drop regime and min-selection can't dodge dropout (dropped candidates would
    always lose to non-dropped ones on loss) -- and (2) applied identically regardless of `learning`, so the
    no-grad exploration selects noise under the exact regime the final recompute trains with.

    Args:
        model_forward (callable): Model forward function.
        loss_calc_wrapper (callable): Wrapper function to compute loss. Accepts:
            (model_forward, conditions, gt_samples, learning, [optional] rand_inputs, [optional] rand_seeds).
            Returns (losses, [optional] predictions).
        conditions (torch.Tensor or tuple): Conditions serving as input to the model.
        gt_samples (torch.Tensor): Ground truth samples.
        best_of_k (int): Number of candidates to explore to find the lowest loss mode.
        max_chunk_bs_mult (int): Max batch size multiplier for each chunk (chunks are looped over, not batches).
            Helps with parallelization.
        save_mem_mode (bool, optional): If True, minimizes gradients during the exploration loop to save memory,
            then recomputes losses with full gradients for the best candidates at the end.
            May result in slightly different values due to nondeterminism. Defaults to True.
        debug_save_mem_mode (bool, optional): If True, verifies that recomputed tensors in save_mem_mode
            are approximately the same as the originals. Not recommended for production as tensors can vary slightly due to nondeterminism and cause errors even when they are relatively close enough.
            Defaults to False.
        not_training (bool, optional): If True, means that the grad will never be tracked as the model is not being trained. Useful for val/testing

    Returns:
        tuple:
            - best_losses (torch.Tensor): The minimum losses found for each batch element.
            - best_predictions (torch.Tensor): The predictions corresponding to the minimum losses.
    """
    
    # short-circuit for best_of_k=1: skip chunking overhead
    if best_of_k == 1:
        learning_direct = not not_training
        rand_inputs = torch.randn_like(gt_samples)
        rand_seeds = torch.randint(0, 2147483647, (gt_samples.shape[0],), device=gt_samples.device)
        losses, predictions = loss_calc_wrapper(model_forward, conditions, gt_samples, learning=learning_direct, rand_inputs=rand_inputs, rand_seeds=rand_seeds, **loss_calc_kwargs)
        return losses, predictions

    # prepare chunking variables ------------------------------------
    if debug_save_mem_mode:
        assert save_mem_mode, "debug_mode can only be used when debug_save_mem_mode is set"
    if not save_mem_mode and not not_training:
        assert max_chunk_bs_mult >= best_of_k, "save_mem_mode=False requires max_chunk_bs_mult >= best_of_k so all candidates fit in a single chunk (otherwise in-place updates to best_losses corrupt the autograd graph). use save_mem_mode=True or increase max_chunk_bs_mult"
    regular_bs = gt_samples.shape[0] # this is the regular batch sizes used for training models; we refer to this as B
    assert max_chunk_bs_mult >= 1, "need to be exploring with a max_chunk_bs_mult >= 1"
    assert best_of_k >=1, "best_of_k needs to be >= 1 for this to work"
    first_iter = True
    total_exploration_bs = regular_bs * best_of_k
    max_chunk_bs = max_chunk_bs_mult * regular_bs
    for_loop_iters = math.ceil(total_exploration_bs / max_chunk_bs)
    remaining_bs = total_exploration_bs
    learning = not save_mem_mode if not not_training else False
    best_predictions = None

    with torch.set_grad_enabled(learning):
        for _ in range(for_loop_iters):
            # prepare per iteration chunking ------------------------------------
            curr_chunk_bs = remaining_bs if remaining_bs <= max_chunk_bs else max_chunk_bs
            assert curr_chunk_bs % regular_bs == 0, "need to use a chunk thats divisible by reg bs, error occurred, please investigate"
            remaining_bs = remaining_bs - curr_chunk_bs
            this_chunk_bs_mult = int(curr_chunk_bs / regular_bs) # we refer to this_chunk_bs_mult as C_BS

            # prepare random conditions ------------------------------------
            rand_inputs = torch.randn((curr_chunk_bs, *gt_samples.shape[1:]), device=gt_samples.device) # C_BS, *gt_shape
            rand_seeds = torch.randint(0, 2147483647, (curr_chunk_bs,), device=gt_samples.device) # C_BS; we use the max 32 bit int value here

            # prepare conditions and ground truth ------------------------------------
            if isinstance(conditions, tuple):
                conditions_expanded = tuple(torch.cat([c] * this_chunk_bs_mult, dim=0) for c in conditions)
            else:
                conditions_expanded = torch.cat([conditions] * this_chunk_bs_mult, dim=0)
            gt_samples_expanded = torch.cat([gt_samples] * this_chunk_bs_mult, dim=0) # C_BS, *gt_shape

            # compute losses and possibly predictions ------------------------------------
            losses, predictions = loss_calc_wrapper(model_forward, conditions_expanded, gt_samples_expanded, learning=learning, rand_inputs=rand_inputs, rand_seeds=rand_seeds, **loss_calc_kwargs)

            # do best of k along chunk ------------------------------------
            chunk_losses_reshaped = losses.reshape(this_chunk_bs_mult, regular_bs) # C_BS, B
            chunk_min_losses, chunk_min_indices = chunk_losses_reshaped.min(dim=0) # (B,), (B,)

            # Convert chunk_min_indices (0..M-1) to flat indices (0..M*B-1), select the best candidate from the current chunk for each batch element
            flat_indices = chunk_min_indices * regular_bs + torch.arange(regular_bs, device=gt_samples.device)

            chunk_best_rand_inputs = rand_inputs[flat_indices]
            chunk_best_rand_seeds = rand_seeds[flat_indices]

            save_best_predictions = (debug_save_mem_mode and predictions is not None) if save_mem_mode else (predictions is not None)

            if save_best_predictions:
                chunk_best_predictions = predictions[flat_indices]

            if first_iter: 
                first_iter = False
                best_rand_inputs = chunk_best_rand_inputs # B, *gt_shape
                best_rand_seeds = chunk_best_rand_seeds # B, 
                best_losses = chunk_min_losses # B, 
                if save_best_predictions:
                    best_predictions = chunk_best_predictions # B, *gt_shape
            else:
                # Update global bests if current chunk found better solutions
                replacement_mask = chunk_min_losses < best_losses
                if replacement_mask.any():
                    best_rand_inputs[replacement_mask] = chunk_best_rand_inputs[replacement_mask] # B, *gt_shape
                    best_rand_seeds[replacement_mask] = chunk_best_rand_seeds[replacement_mask] # B, 
                    best_losses[replacement_mask] = chunk_min_losses[replacement_mask] # B, 
                    if save_best_predictions:
                        best_predictions[replacement_mask] = chunk_best_predictions[replacement_mask] # B, *gt_shape
    
    # finished for loop, if save_mem_mode do last forward, else return best ------------------------------------
    if save_mem_mode:
        learning = True if not not_training else False
        torch.clear_autocast_cache() # clear cached bf16 parameter copies from the no-grad exploration loop so the recomputation builds a fresh autograd graph
        final_losses, final_predictions = loss_calc_wrapper(model_forward, conditions, gt_samples, learning=learning, rand_inputs=best_rand_inputs, rand_seeds=best_rand_seeds, **loss_calc_kwargs) # set learning to True
        
        if debug_save_mem_mode:
            if best_predictions is not None:
                assert torch.allclose(best_predictions, final_predictions, rtol=1e-5, atol=1e-8), "predictions did not reproduce when doing 2nd round for comp graph"
            assert torch.allclose(best_losses, final_losses, rtol=1e-5, atol=1e-8), "losses did not reproduce when doing 2nd round for comp graph"

        return final_losses, final_predictions

    else: # already computed best_losses and best_predictions
        return best_losses, best_predictions


# most of below 3 funcs are from vjepa2 codebase https://github.com/facebookresearch/vjepa2
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate to
        # [2*lower-1, 2*upper-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b) # type: (Tensor, float, float, float, float) -> Tensor


def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)


