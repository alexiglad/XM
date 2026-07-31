import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import copy

from model.model_utils import *
from model.jumpy import JumpyGenerativeModel


class Jumpy_IMG_CC(L.LightningModule):
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

        self.feed_forward_transformer = setup_diffusion_transformer(self.hparams) # use diffusion transformer with time set to default value

        if self.hparams.final_layer_init_std > 0:
            from model.model_utils import init_dit_final_layer
            init_dit_final_layer(self.feed_forward_transformer, self.hparams.final_layer_init_std)

        self.label_embedder = LabelEmbedder(
            num_classes=self.num_classes,
            hidden_size=self.hparams.embedding_dim,
            dropout_prob=self.cfg_dropout_prob
        )
        nn.init.normal_(self.label_embedder.embedding_table.weight, std=0.02) # based on dit paper

        self.jumpy = JumpyGenerativeModel(
            num_jumps=self.hparams.jumpy_n_jumps,
            pred_type=self.hparams.jumpy_pred_type,
            num_inference_jumps=self.hparams.jumpy_inference_steps,
            time_conditioning=self.hparams.jumpy_time_conditioning,
            time_embed_dim=None, # DiT has its own t_embedder
            time_embed_cont=self.hparams.jumpy_time_embed_cont,
            loss_reweighting=self.hparams.jumpy_loss_reweighting,
        )

        self.reset_image_encoder_decoder = False # to fix a bug with loading sdxl vae
        self.finished_warming_up = False

        if self.hparams.ema_model != 1.0: #NOTE need to add all learnable params here
            self.ema_feed_forward_transformer = copy.deepcopy(self.feed_forward_transformer)
            self.ema_label_embedder = copy.deepcopy(self.label_embedder)

            self.regular_model_submodules = [self.feed_forward_transformer, self.label_embedder]
            self.ema_model_submodules = [self.ema_feed_forward_transformer, self.ema_label_embedder]

            check_model_ema_params(self)

    def forward(self, x, y=None, learning=False, use_ema=False, force_drop_ids=None, **kwargs): # y is the class labels; deterministic so random_seeds not used
        with torch.set_grad_enabled(learning):
            t_jump = kwargs.get('t', None)
            t = t_jump.float() if t_jump is not None else torch.zeros((x.shape[0],), device=self.device) # dit t_embedder handles jump index

            label_embedder = self.ema_label_embedder if use_ema else self.label_embedder
            feed_forward_transformer = self.ema_feed_forward_transformer if use_ema else self.feed_forward_transformer

            class_embeddings = label_embedder(y, learning, force_drop_ids=force_drop_ids) # B, D
            return feed_forward_transformer(x, t, y=class_embeddings) # B, C, W, H

    def loss_calc_func(self, predicted_embeddings, image_embeddings, reduction="mean"):
        losses = F.mse_loss(predicted_embeddings, image_embeddings, reduction=reduction)

        if reduction == "mean":
            return losses
        else:
            return losses.mean(dim=tuple(range(1, losses.ndim)))

    def loss_calc_wrapper(self, model_forward, conditions, gt_samples, learning=True, rand_inputs=None, rand_seeds=None, jump_k=None, cfg_drop_mask=None):
        # CFG dropout: tile the per-sample cfg_drop_mask across candidates (see the CFG nuance NOTE in xm_chunked_best_of_k)
        # always pass force_drop_ids (even all-zero) so LabelEmbedder doesnt re-sample dropout internally with a different decision
        force_drop_ids = None
        if cfg_drop_mask is not None:
            chunk_mult = conditions.shape[0] // cfg_drop_mask.shape[0]
            force_drop_ids = cfg_drop_mask.repeat(chunk_mult).long()

        with torch.set_grad_enabled(learning):
            t = None

            chunk_mult = rand_inputs.shape[0] // jump_k.shape[0]
            k = jump_k.repeat(chunk_mult)
            effective_input, effective_gt = self.jumpy.get_input_and_target(rand_inputs, gt_samples, k)
            if self.hparams.jumpy_time_conditioning:
                t = self.jumpy.k_to_time(k)

            pred_samples = model_forward(effective_input, y=conditions, t=t, learning=learning, force_drop_ids=force_drop_ids)
            loss = self.loss_calc_func(pred_samples, effective_gt, reduction="none")
            loss = self.jumpy.apply_loss_weight(loss, k)
            return loss, None

    def forward_loss_wrapper(self, x, phase="train"):
        class_labels = x['label']
        image_embeddings = x['latent'] if 'latent' in x else get_encoded_images(x['image'], self.hparams.backbone_type, self.image_encoder, sdxl_vae_standardization=True) # B, C, W, H
        learning = (phase == "train")

        jump_k = self.jumpy.sample_timestep(image_embeddings.shape[0], image_embeddings.device) # if jumpy_n_jumps = 1 then this will be all zeros

        cfg_drop_mask = None
        if learning and self.cfg_dropout_prob > 0:
            cfg_drop_mask = torch.rand(image_embeddings.shape[0], device=image_embeddings.device) < self.cfg_dropout_prob

        loss, _ = xm_chunked_best_of_k(self.forward, self.loss_calc_wrapper, class_labels, image_embeddings, self.hparams.xm_best_of_k, self.hparams.xm_chunk_bs_mult, save_mem_mode=self.hparams.xm_save_mem_mode, debug_save_mem_mode=self.hparams.xm_debug_mode, not_training=not learning, jump_k=jump_k, cfg_drop_mask=cfg_drop_mask)

        log_dict = {
            'loss': loss.mean(),
        }

        def generate_fn():
            z = torch.randn_like(image_embeddings[:1])
            generated = self.generate_samples(z, class_labels[:1])
            return {'generated_image': generated[0], 'original_image': image_embeddings[0]}
        log_generated_samples(self, phase, log_dict, generate_fn)

        return log_dict

    def generate_samples(self, z, y, cfg_scale=None, use_ema=False):
        def model_fn(x, t=None):
            if cfg_scale and cfg_scale > 0.0:
                out_c = self.forward(x, y=y, t=t, learning=False, use_ema=use_ema, force_drop_ids=None)
                out_u = self.forward(x, y=y, t=t, learning=False, use_ema=use_ema, force_drop_ids=torch.ones_like(y))
                return out_u + cfg_scale * (out_c - out_u)
            return self.forward(x, y=y, t=t, learning=False, use_ema=use_ema)
        return self.jumpy.inference(z, model_fn)

    def ema_update(self):
        update_ema_model(self.regular_model_submodules, self.ema_model_submodules, self.hparams.ema_model)