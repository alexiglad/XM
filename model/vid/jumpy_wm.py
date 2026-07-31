import random
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import copy

from model.model_utils import *
from model.jumpy import JumpyGenerativeModel


class Jumpy_VID_WM(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        self.image_encoder = load_image_encoder(self.hparams.backbone_type, self.hparams.vit_backbone_size, device=self.device, use_ema=True)
        
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.cfg_dropout_prob = self.hparams.cfg_dropout_prob

        assert self.hparams.video_transformer_3d, "this wm code only supports 3d transformers for now"
        assert not self.hparams.cross_attention_transformer, "this code doesnt support cross_attention_transformer as of now"
        self.feed_forward_transformer = setup_3d_transformer(self.hparams, use_adaln=self.hparams.jumpy_time_conditioning)

        if self.hparams.backbone_type == "vae":
            self.encoder_dim = 4 * self.hparams.image_dims[0]**2 // 64
            print("self.encoder_dim", self.encoder_dim)
            self.image_cwh = (3, self.hparams.image_dims[0], self.hparams.image_dims[1])
            print("self.image_cwh", self.image_cwh)
            self.embed_cwh = (4, self.hparams.image_dims[0] // 8, self.hparams.image_dims[0] // 8)
            assert self.hparams.patch_size <= self.hparams.image_dims[0] // 8, "patch size set by hparams must be <= raw patch size"
            print("self.embed_cwh", self.embed_cwh)
        elif self.hparams.backbone_type == "rae":
            # RAE with DINOv2: always encodes to 16x16 spatial grid, 768 channels
            self.encoder_dim = 768 * 16 * 16
            print("self.encoder_dim", self.encoder_dim)
            self.image_cwh = (3, self.hparams.image_dims[0], self.hparams.image_dims[1])
            print("self.image_cwh", self.image_cwh)
            self.embed_cwh = (768, 16, 16)
            assert self.hparams.patch_size <= 16, "patch size set by hparams must be <= rae spatial size (16)"
            print("self.embed_cwh", self.embed_cwh)

        else:
            raise ValueError(f"encoder_dim unknown for self.hparams.backbone_type: {self.hparams.backbone_type}, please add support")

        self.jumpy = JumpyGenerativeModel(
            num_jumps=self.hparams.jumpy_n_jumps,
            pred_type=self.hparams.jumpy_pred_type,
            num_inference_jumps=self.hparams.jumpy_inference_steps,
            time_conditioning=self.hparams.jumpy_time_conditioning,
            time_embed_weight_init=self.hparams.weight_initialization_method,
            time_embed_dim=None, # use transformer's built-in t_embedder (via adaln) instead of a separate one
            time_embed_cont=self.hparams.jumpy_time_embed_cont,
            loss_reweighting=self.hparams.jumpy_loss_reweighting,
        )
        if self.hparams.jumpy_time_conditioning:
            assert self.hparams.jumpy_time_embed_cont, "jumpy_wm uses the transformer's built-in sinusoidal t_embedder, so jumpy_time_embed_cont must be True here; discrete int indices 0..K-1 collapse under max_period=10000"

        if not self.hparams.gc_world_model_unconditional:
            n_cond = len(self.hparams.world_modeling_given_frames)
            self.null_cond_embed = nn.Embedding(n_cond, self.encoder_dim)
            nn.init.normal_(self.null_cond_embed.weight, std=0.02)

        self.reset_image_encoder_decoder = False
        self.finished_warming_up = False
        self.log_val_video = False

        if self.hparams.ema_model != 1.0: #NOTE need to add all learnable params here
            self.ema_feed_forward_transformer = copy.deepcopy(self.feed_forward_transformer)
            self.regular_model_submodules = [self.feed_forward_transformer]
            self.ema_model_submodules = [self.ema_feed_forward_transformer]
            if not self.hparams.gc_world_model_unconditional:
                self.ema_null_cond_embed = copy.deepcopy(self.null_cond_embed)
                self.regular_model_submodules.append(self.null_cond_embed)
                self.ema_model_submodules.append(self.ema_null_cond_embed)
            check_model_ema_params(self)

    def _get_idx(self, device):
        # in unconditional mode, treat world_modeling_given_frames as empty so every position is a prediction
        if self.hparams.gc_world_model_unconditional:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.as_tensor(self.hparams.world_modeling_given_frames, device=device)

    def forward(self, x, y=None, learning=False, use_ema=False, force_drop_ids=None, **kwargs): # deterministic so random_seeds not used
        # x = B, S, D; y = B, n_frames, D if is there
        with torch.set_grad_enabled(learning):
            t = kwargs.get('t', None)
            idx = self._get_idx(x.device)
            feed_forward_transformer = self.ema_feed_forward_transformer if use_ema else self.feed_forward_transformer

            if y is not None:
                x = x.clone()
                if force_drop_ids is not None:
                    null_cond_embed = self.ema_null_cond_embed if use_ema else self.null_cond_embed
                    null_y = null_cond_embed(torch.arange(y.shape[1], device=x.device)) # n_cond, D
                    y = y.clone()
                    drop = force_drop_ids == 1
                    y[drop] = null_y
                x[:, idx] = y
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            x = x.reshape(batch_size, seq_len, *self.embed_cwh) # B, S, C, W, H
            x = x.permute(dims=(0, 2, 1, 3, 4)) # B, C, S, W, H
            preds = feed_forward_transformer(x, t=t) # B, C, S, W, H; t passed to transformer's built-in t_embedder for adaln
            preds = preds.permute(dims=(0, 2, 1, 3, 4)) # B, S, C, W, H
            preds = preds.reshape(batch_size, seq_len, self.encoder_dim) # B, S, D
            return preds

    def loss_calc_func(self, predicted_embeddings, gt_embeddings, reduction="mean"):
        idx = self._get_idx(predicted_embeddings.device)
        seq_mask = torch.ones(predicted_embeddings.size(1), dtype=torch.bool, device=predicted_embeddings.device)
        seq_mask[idx] = False
        predicted_embeddings = predicted_embeddings[:, seq_mask] # B, S'
        gt_embeddings = gt_embeddings[:, seq_mask] # B, S'

        losses = F.mse_loss(predicted_embeddings, gt_embeddings, reduction=reduction)

        if reduction == "mean":
            return losses
        else:
            return losses.mean(dim=tuple(range(1, losses.ndim)))

    def jumpy_inference(self, start_x, world_model_conditioning, use_ema=False, cfg_scale=0.0):
        """Run jumpy.num_inference_jumps steps. Works for jumpy_n_jumps=1 too aka e2e prediction."""
        idx = self._get_idx(start_x.device)
        if self.hparams.gc_world_model_unconditional:
            world_model_conditioning = world_model_conditioning[:, :0] # drop conditioning slots so forward's x[:, idx] = y becomes a no-op
        def model_fn(x, t=None):
            if cfg_scale > 0.0 and not self.hparams.gc_world_model_unconditional:
                null_cond_embed = self.ema_null_cond_embed if use_ema else self.null_cond_embed
                null_y = null_cond_embed(torch.arange(world_model_conditioning.shape[1], device=x.device))
                null_y = null_y.unsqueeze(0).expand(x.shape[0], -1, -1)
                out_c = self.forward(x, y=world_model_conditioning, t=t, learning=False, use_ema=use_ema)
                out_u = self.forward(x, y=null_y, t=t, learning=False, use_ema=use_ema)
                return out_u + cfg_scale * (out_c - out_u)
            return self.forward(x, y=world_model_conditioning, t=t, learning=False, use_ema=use_ema)
        def post_step(x): # used for stuff done in between steps e.g. masking, unmasking, etc
            x = x.clone()
            x[:, idx] = world_model_conditioning
            return x
        return self.jumpy.inference(start_x, model_fn, post_step_fn=post_step)

    def loss_calc_wrapper(self, model_forward, conditions, gt_samples, learning=True, rand_inputs=None, rand_seeds=None, jump_k=None, cfg_drop_mask=None):
        # CFG dropout: tile the per-sample cfg_drop_mask across candidates (see the CFG nuance NOTE in xm_chunked_best_of_k)
        # never short-circuit on cfg_drop_mask.any(): masks differ across DDP ranks, and skipping null_cond_embed on only some ranks mismatches gradient buckets (find_unused_parameters=False hangs)
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
            recon_loss = self.loss_calc_func(pred_samples, effective_gt, reduction="none")
            recon_loss = self.jumpy.apply_loss_weight(recon_loss, k)
            return recon_loss, None
    
    def forward_loss_wrapper(self, x, phase="train"):
        x = x['frames']
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        assert self.hparams.gc_world_model_unconditional or len(self.hparams.world_modeling_given_frames) > 0, "need sufficient world model frame conditioning (or set --gc_world_model_unconditional)"

        x = x.reshape(-1, *x.shape[2:]) # B*S, C, W, H
        if not self.hparams.preencode_dataset:
            gt_video_embeddings = get_encoded_images(x, self.hparams.backbone_type, self.image_encoder, sdxl_vae_standardization = self.hparams.sdxl_vae_standardization)
        else:
            gt_video_embeddings = x
        
        gt_video_embeddings = gt_video_embeddings.reshape(batch_size, seq_length, self.encoder_dim) # B, S, D

        # NOTE masked_video_embeddings includes some image embeddings that are unmasked, based on world_modeling_given_frames
        idx = self._get_idx(gt_video_embeddings.device)
        world_model_conditioning = gt_video_embeddings[:, idx].clone() # B, len(world_modeling_given_frames), D (empty if uncond)

        learning = (phase == "train")
        jump_k = self.jumpy.sample_timestep(gt_video_embeddings.shape[0], gt_video_embeddings.device) # if jumpy_n_jumps = 1 then this will be all zeros

        cfg_drop_mask = None
        if learning and self.cfg_dropout_prob > 0 and not self.hparams.gc_world_model_unconditional:
            cfg_drop_mask = torch.rand(batch_size, device=gt_video_embeddings.device) < self.cfg_dropout_prob

        loss, _ = xm_chunked_best_of_k(self.forward, self.loss_calc_wrapper, world_model_conditioning, gt_video_embeddings, self.hparams.xm_best_of_k, self.hparams.xm_chunk_bs_mult, save_mem_mode=self.hparams.xm_save_mem_mode, debug_save_mem_mode=self.hparams.xm_debug_mode, not_training=not learning, jump_k=jump_k, cfg_drop_mask=cfg_drop_mask)

        log_dict = {
            'loss': loss.mean(),
        }

        def generate_fn():
            bi = 0 if phase == "train" else random.randint(0, batch_size - 1)
            masked = torch.randn_like(gt_video_embeddings)
            masked[:, idx] = gt_video_embeddings[:, idx].clone()
            noise_input = masked[bi].unsqueeze(0)
            generated = self.generate_samples(noise_input, world_model_conditioning[bi:bi+1])
            return {
                'generated_video': generated[0].reshape(seq_length, *self.embed_cwh),
                'gt_video': gt_video_embeddings[bi].reshape(seq_length, *self.embed_cwh),
                'masked_video': masked[bi].reshape(seq_length, *self.embed_cwh),
            }
        log_generated_samples(self, phase, log_dict, generate_fn, log_during_val=True)

        return log_dict

    def generate_samples(self, noise, world_model_conditioning, use_ema=False, cfg_scale=0.0, **kwargs):
        return self.jumpy_inference(noise, world_model_conditioning, use_ema=use_ema, cfg_scale=cfg_scale or 0.0)

    def ema_update(self):
        update_ema_model(self.regular_model_submodules, self.ema_model_submodules, self.hparams.ema_model)
