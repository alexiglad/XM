import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy
import traceback
from torchvision.transforms import functional as TF
import torchvision.models as models
import math
import random
from diffusers import AutoencoderKL
import copy

from model.model_utils import *
from model.diffusion import create_diffusion
from model.diffusion.gaussian_diffusion import mean_flat
from model.flow import create_flow

class DiT_IMG_CC(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        self.image_encoder = load_image_encoder(self.hparams.backbone_type, self.hparams.vit_backbone_size, device=self.device, use_ema=True)
        
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.num_classes = self.hparams.num_classes if self.hparams.num_classes > 0 else 1000
        self.cfg_dropout_prob = self.hparams.cfg_dropout_prob  # for cfg

        self.diffusion_transformer = setup_diffusion_transformer(self.hparams)

        self.label_embedder = LabelEmbedder(
            num_classes=self.num_classes,
            hidden_size=self.hparams.embedding_dim,
            dropout_prob=self.cfg_dropout_prob
        )
        nn.init.normal_(self.label_embedder.embedding_table.weight, std=0.02) # based on dit paper
        
        if self.hparams.diffusion_supervision_type == 'score':
            self.diffusion = create_diffusion(
                timestep_respacing="", 
                diffusion_steps=self.hparams.diffusion_steps,
                learn_sigma=self.hparams.learn_sigma, # use learn_sigma = False for most experiments since it maximizes generation based on iddpm and is more comparable to EBM
            )
        else: # velocity diffusion_supervision_type
            self.diffusion = create_flow(ode_method=self.hparams.ode_method, ode_step_size=self.hparams.ode_step_size)

        if self.hparams.use_ot_flow:
            from model.flow.ot_coupling import OTPlanSampler
            self.ot_sampler = OTPlanSampler(method="exact")

        self.reset_image_encoder_decoder = False
        self.finished_warming_up = False

        if self.hparams.ema_model != 1.0: #NOTE need to add all learnable params here
            self.ema_diffusion_transformer = copy.deepcopy(self.diffusion_transformer)
            self.ema_label_embedder = copy.deepcopy(self.label_embedder)

            self.regular_model_submodules = [self.diffusion_transformer, self.label_embedder]
            self.ema_model_submodules = [self.ema_diffusion_transformer, self.ema_label_embedder]

            check_model_ema_params(self)

    def forward(self, x, t, y=None, learning=False, use_ema=False, force_drop_ids=None, **kwargs): # y is the class labels
        label_embedder = self.ema_label_embedder if use_ema else self.label_embedder
        diffusion_transformer = self.ema_diffusion_transformer if use_ema else self.diffusion_transformer

        class_embeddings = label_embedder(y, learning, force_drop_ids=force_drop_ids) # B, D
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return diffusion_transformer(x, t, y=class_embeddings) # B, C, W, H


    def loss_calc_wrapper(self, model_forward, conditions, gt_samples, learning=True, rand_inputs=None, rand_seeds=None, cfg_drop_mask=None):
        # NOTE the reason this is necessary on top of training_losses is we need to do model_kwargs and forward wont accept the conditions in the right order by default if we just pass all conditions to training_losses from xm_chunked_best_of_k
        # CFG dropout: tile the per-sample cfg_drop_mask across candidates (see the CFG nuance NOTE in xm_chunked_best_of_k)
        # always pass force_drop_ids (even all-zero) so LabelEmbedder doesnt re-sample dropout internally with a different decision
        force_drop_ids = None
        if cfg_drop_mask is not None:
            chunk_mult = conditions[1].shape[0] // cfg_drop_mask.shape[0]
            force_drop_ids = cfg_drop_mask.repeat(chunk_mult).long()

        with torch.set_grad_enabled(learning):
            model_kwargs=dict(y=conditions[1], learning=learning, force_drop_ids=force_drop_ids)
            terms = self.diffusion.training_losses(model_forward, gt_samples, conditions[0], model_kwargs, noise=rand_inputs, reduce_loss=False)
            return terms['loss'], None # return tuple here since thats what xm_chunked_best_of_k expects

    
    def forward_loss_wrapper(self, x, phase="train"):
        class_labels = x['label']
        image_embeddings = x['latent'] if 'latent' in x else get_encoded_images(x['image'], self.hparams.backbone_type, self.image_encoder, sdxl_vae_standardization=True) # B, C, W, H
        learning = (phase == "train")

        if self.hparams.diffusion_supervision_type == 'score':
            t = torch.randint(0, self.diffusion.num_timesteps, (image_embeddings.shape[0],), device=self.device)
        else: #  velocity diffusion_supervision_type
            t = torch.rand(image_embeddings.shape[0], device=self.device)

        if self.hparams.xm_best_of_k <= 1: # no exploration, default dit
            noise = None
            if self.hparams.use_ot_flow:
                # OT-CFM: pair noise with data via minibatch EMD plan, sampled with replacement (torchcfm default)
                noise = torch.randn_like(image_embeddings)
                noise, image_embeddings, class_labels = self.ot_sampler.sample_plan_with_labels(noise, image_embeddings, y1=class_labels)
            model_kwargs = dict(y=class_labels, learning=learning)
            log_dict = self.diffusion.training_losses(self.forward, image_embeddings, t, model_kwargs, noise=noise)

            key_map = {"vb": "vb_loss", "mse": "score_pred_loss"}
            log_dict = {key_map.get(k, k): (v.mean().detach() if k != 'loss' else v.mean()) for k, v in log_dict.items()} # detach all non loss keys

        # Explorative Modeling changes below ############
        else:
            cfg_drop_mask = None
            if learning and self.cfg_dropout_prob > 0:
                cfg_drop_mask = torch.rand(image_embeddings.shape[0], device=self.device) < self.cfg_dropout_prob

            conditions = (t, class_labels)
            loss, __ = xm_chunked_best_of_k(self.forward, self.loss_calc_wrapper, conditions, image_embeddings, self.hparams.xm_best_of_k, self.hparams.xm_chunk_bs_mult, save_mem_mode=self.hparams.xm_save_mem_mode, debug_save_mem_mode=self.hparams.xm_debug_mode, not_training=not learning, cfg_drop_mask=cfg_drop_mask)
            log_dict = {"loss": loss.mean()}

        def generate_fn():
            z = torch.randn_like(image_embeddings[0].unsqueeze(0))
            samples = self.generate_samples(z, class_labels[0].unsqueeze(0))
            return {'generated_image': samples[0], 'original_image': image_embeddings[0]}
        log_generated_samples(self, phase, log_dict, generate_fn)

        return log_dict

    def generate_samples(self, z, y, cfg_scale=None, use_ema=False):
        def model_fn(x, t, y=y):
            if cfg_scale and cfg_scale > 0.0:
                out_c = self.forward(x, t, y=y, learning=False, use_ema=use_ema)
                out_u = self.forward(x, t, y=y, learning=False, use_ema=use_ema, force_drop_ids=torch.ones_like(y))
                return out_u + cfg_scale * (out_c - out_u)
            return self.forward(x, t, y=y, learning=False, use_ema=use_ema)
        return self.diffusion.p_sample_loop(model_fn, z.shape, z, clip_denoised=False, model_kwargs=dict(y=y), device=z.device, progress=False)

    def ema_update(self):
        update_ema_model(self.regular_model_submodules, self.ema_model_submodules, self.hparams.ema_model)