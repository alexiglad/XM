import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import copy
import random

from model.model_utils import *
from model.diffusion import create_diffusion
from model.flow import create_flow


class DiT_VID_WM(L.LightningModule):
    """
    Diffusion Transformer World Model for video prediction.
    Predicts an entire video at once using diffusion.
    - Time conditioning (diffusion timestep) via adaLN in the 3D ViT backbone.
    - Video conditioning (given frames) is context provided via attention.
    """
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        self.image_encoder = load_image_encoder(self.hparams.backbone_type, self.hparams.vit_backbone_size, device=self.device, use_ema=True)
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        if self.hparams.backbone_type == "vae":
            self.encoder_dim = 4 * self.hparams.image_dims[0]**2 // 64
            self.image_cwh = (3, self.hparams.image_dims[0], self.hparams.image_dims[1])
            self.embed_cwh = (4, self.hparams.image_dims[0] // 8, self.hparams.image_dims[0] // 8)
            assert self.hparams.patch_size <= self.hparams.image_dims[0] // 8, "patch size must be <= raw patch size"
        elif self.hparams.backbone_type == "rae":
            self.encoder_dim = 768 * 16 * 16
            self.image_cwh = (3, self.hparams.image_dims[0], self.hparams.image_dims[1])
            self.embed_cwh = (768, 16, 16)
            assert self.hparams.patch_size <= 16, "patch size must be <= rae spatial size (16)"
        else:
            raise ValueError(f"encoder_dim unknown for backbone_type: {self.hparams.backbone_type}")

        # 3D Vision Transformer with adaLN for time conditioning
        self.diffusion_transformer = setup_3d_transformer(self.hparams, is_energy=False, use_adaln=True)

        self.cfg_dropout_prob = self.hparams.cfg_dropout_prob

        C, H, W = self.embed_cwh
        if not self.hparams.gc_world_model_unconditional:
            n_cond = len(self.hparams.world_modeling_given_frames)
            self.null_cond_embed = nn.Embedding(n_cond, C * H * W)
            nn.init.normal_(self.null_cond_embed.weight, std=0.02)

        if self.hparams.diffusion_supervision_type == 'score':
            assert not self.hparams.learn_sigma, "learn_sigma is not supported for video world model generation (PatchUnembedder outputs in_chans, not 2*in_chans)"
            self.diffusion = create_diffusion(
                timestep_respacing="",
                diffusion_steps=self.hparams.diffusion_steps,
                learn_sigma=self.hparams.learn_sigma,
            )
        else:  # velocity / flow matching
            self.diffusion = create_flow(ode_method=self.hparams.ode_method, ode_step_size=self.hparams.ode_step_size)

        self.reset_image_encoder_decoder = False
        self.finished_warming_up = False
        self.log_val_video = False

        if self.hparams.ema_model != 1.0:
            self.ema_diffusion_transformer = copy.deepcopy(self.diffusion_transformer)
            self.regular_model_submodules = [self.diffusion_transformer]
            self.ema_model_submodules = [self.ema_diffusion_transformer]
            if not self.hparams.gc_world_model_unconditional:
                self.ema_null_cond_embed = copy.deepcopy(self.null_cond_embed)
                self.regular_model_submodules.append(self.null_cond_embed)
                self.ema_model_submodules.append(self.ema_null_cond_embed)
            check_model_ema_params(self)

    def _get_null_cond_5d(self, n_cond, device, use_ema=False):
        """Get learned null conditioning in 5D format: (1, C, n_cond, H, W)."""
        null_cond_embed = self.ema_null_cond_embed if use_ema else self.null_cond_embed
        C, H, W = self.embed_cwh
        return null_cond_embed(torch.arange(n_cond, device=device)).reshape(1, n_cond, C, H, W).permute(0, 2, 1, 3, 4)

    def forward(self, x_pred_noisy, t, y=None, learning=False, use_ema=False, **kwargs):
        """
        x_pred_noisy : (B, C, n_pred, H, W) noisy prediction frames
        y            : (B, C, n_cond, H, W) conditioning frames; None = unconditional
        Returns      : (B, C, n_pred, H, W) predicted noise or velocity at prediction positions
        """
        diffusion_transformer = self.ema_diffusion_transformer if use_ema else self.diffusion_transformer
        B, C, n_pred, H, W = x_pred_noisy.shape
        device = x_pred_noisy.device

        idx = self._get_idx(device)
        seq_length = n_pred + idx.shape[0]
        pred_mask = self._get_pred_mask(seq_length, device)

        # Reconstruct full video: noisy pred frames + (dropped or clean) cond frames
        x_full = torch.empty(B, C, seq_length, H, W, device=device, dtype=x_pred_noisy.dtype)
        x_full[:, :, pred_mask, :, :] = x_pred_noisy
        if y is not None:
            x_full[:, :, idx, :, :] = y
        else:
            x_full[:, :, idx, :, :] = torch.randn(B, C, idx.shape[0], H, W, device=device, dtype=x_pred_noisy.dtype)

        if t.ndim == 0:
            t = t.expand(B)
        elif t.shape[0] == 1 and B > 1:
            t = t.expand(B)

        out_full = diffusion_transformer(x_full, t=t)
        return out_full[:, :, pred_mask, :, :]

    def _get_idx(self, device):
        # in unconditional mode, treat world_modeling_given_frames as empty so every position is a prediction
        if self.hparams.gc_world_model_unconditional:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.as_tensor(self.hparams.world_modeling_given_frames, device=device)

    def _get_pred_mask(self, seq_len, device):
        idx = self._get_idx(device)
        pred_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        pred_mask[idx] = False
        return pred_mask

    def loss_calc_wrapper(self, model_forward, conditions, gt_samples, learning=True, rand_inputs=None, rand_seeds=None, diffusion_t=None, cfg_drop_mask=None):
        # CFG dropout: tile the per-sample cfg_drop_mask across candidates (see the CFG nuance NOTE in xm_chunked_best_of_k)
        # never short-circuit on cfg_drop_mask.any(): masks differ across DDP ranks, and skipping null_cond_embed on only some ranks mismatches gradient buckets (find_unused_parameters=False hangs)
        if cfg_drop_mask is not None:
            chunk_mult = conditions.shape[0] // cfg_drop_mask.shape[0]
            drop = cfg_drop_mask.repeat(chunk_mult)
            null_y = self._get_null_cond_5d(conditions.shape[2], conditions.device) # 1, C, n_cond, H, W
            conditions = conditions.clone()
            conditions[drop] = null_y

        with torch.set_grad_enabled(learning):
            chunk_mult = rand_inputs.shape[0] // diffusion_t.shape[0]
            t = diffusion_t.repeat(chunk_mult)

            model_kwargs = dict(y=conditions, learning=learning)
            if self.hparams.diffusion_supervision_type == 'score':
                losses_dict = self.diffusion.training_losses(model_forward, gt_samples, t, model_kwargs=model_kwargs, noise=rand_inputs)
            else:
                losses_dict = self.diffusion.training_losses(model_forward, gt_samples, t, model_kwargs=model_kwargs, noise=rand_inputs, reduce_loss=False)
            return losses_dict['loss'], None

    def forward_loss_wrapper(self, x, phase="train"):
        x = x['frames']
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        assert self.hparams.gc_world_model_unconditional or len(self.hparams.world_modeling_given_frames) > 0, "need conditioning frames for world model (or set --gc_world_model_unconditional)"

        x = x.reshape(-1, *x.shape[2:])  # B*S, C, H, W
        if not self.hparams.preencode_dataset:
            gt_embeddings_flat = get_encoded_images(x, self.hparams.backbone_type, self.image_encoder, sdxl_vae_standardization=self.hparams.sdxl_vae_standardization)
        else:
            gt_embeddings_flat = x

        gt_embeddings = gt_embeddings_flat.reshape(batch_size, seq_length, *self.embed_cwh)
        gt_video_5d = gt_embeddings.permute(0, 2, 1, 3, 4)  # B, C, S, H, W

        idx = self._get_idx(gt_video_5d.device)
        pred_mask = self._get_pred_mask(seq_length, gt_video_5d.device)
        cond_5d = gt_video_5d[:, :, idx, :, :]
        pred_5d = gt_video_5d[:, :, pred_mask, :, :]

        learning = (phase == "train")

        if self.hparams.diffusion_supervision_type == 'score':
            t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
        else:
            t = torch.rand(batch_size, device=self.device)

        cfg_drop_mask = None
        if learning and self.cfg_dropout_prob > 0 and not self.hparams.gc_world_model_unconditional:
            cfg_drop_mask = torch.rand(batch_size, device=self.device) < self.cfg_dropout_prob

        loss, _ = xm_chunked_best_of_k(self.forward, self.loss_calc_wrapper, cond_5d, pred_5d, self.hparams.xm_best_of_k, self.hparams.xm_chunk_bs_mult, save_mem_mode=self.hparams.xm_save_mem_mode, debug_save_mem_mode=self.hparams.xm_debug_mode, not_training=not learning, diffusion_t=t, cfg_drop_mask=cfg_drop_mask)
        log_dict = {'loss': loss.mean()}

        def generate_fn():
            bi = 0 if phase == "train" else random.randint(0, batch_size - 1)
            gt_5d = gt_video_5d[bi:bi+1]
            cond_5d = gt_5d[:, :, idx, :, :]
            generated_5d = self.sample(cond_5d, gt_5d.shape, n_steps=self.hparams.fvd_inference_diffusion_steps, device=self.device)
            masked_5d = torch.randn_like(gt_5d)
            masked_5d[:, :, idx, :, :] = cond_5d
            return {
                'generated_video': generated_5d[0].permute(1, 0, 2, 3),  # C,S,H,W -> S,C,H,W
                'gt_video': gt_5d[0].permute(1, 0, 2, 3),
                'masked_video': masked_5d[0].permute(1, 0, 2, 3),
            }
        log_generated_samples(self, phase, log_dict, generate_fn, log_during_val=True)

        return log_dict

    @torch.no_grad()
    def sample(self, cond_5d, shape, n_steps=100, device=None, use_ema=False, cfg_scale=0.0):
        """
        cond_5d   : (B, C, n_cond, H, W) clean conditioning frames
        shape     : full video shape (B, C, T, H, W)
        cfg_scale : classifier-free guidance scale (0.0 = no CFG)
        Returns   : (B, C, T, H, W) generated video with conditioning positions = cond_5d
        """
        if device is None:
            device = cond_5d.device

        if self.hparams.gc_world_model_unconditional:
            cond_5d = cond_5d[:, :, :0] # drop conditioning slots so forward's x_full[:, :, idx, :, :] = y becomes a no-op

        B, C, T, H, W = shape
        idx = self._get_idx(device)
        pred_mask = self._get_pred_mask(T, device)
        pred_shape = (B, C, int(pred_mask.sum()), H, W)

        def _model_fn(x, t, **kwargs):
            if cfg_scale > 0.0 and not self.hparams.gc_world_model_unconditional:
                null_y = self._get_null_cond_5d(cond_5d.shape[2], device, use_ema=use_ema).expand(B, -1, -1, -1, -1)
                v_cond = self.forward(x, t, y=cond_5d, use_ema=use_ema)
                v_uncond = self.forward(x, t, y=null_y, use_ema=use_ema)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            return self.forward(x, t, y=cond_5d, use_ema=use_ema)

        x_T = torch.randn(pred_shape, device=device)

        if self.hparams.diffusion_supervision_type == 'score':
            infer_diffusion = create_diffusion(
                timestep_respacing=str(n_steps),
                diffusion_steps=self.hparams.diffusion_steps,
                learn_sigma=self.hparams.learn_sigma,
            )
            generated_pred = infer_diffusion.p_sample_loop(
                _model_fn, pred_shape, x_T,
                clip_denoised=False, model_kwargs={}, progress=False, device=device
            )
        else:
            infer_flow = create_flow(ode_method=self.hparams.ode_method, ode_step_size=1.0 / n_steps)
            generated_pred = infer_flow.p_sample_loop(
                _model_fn, pred_shape, noise=x_T, model_kwargs={}, device=device
            )

        # Reconstruct full video with clean conditioning frames
        generated = torch.empty(shape, device=device)
        generated[:, :, pred_mask, :, :] = generated_pred
        generated[:, :, idx, :, :] = cond_5d
        return generated

    def generate_samples(self, noise, world_model_conditioning, use_ema=False, cfg_scale=None, n_steps=None):
        B, T, D = noise.shape
        n_steps = n_steps or self.hparams.fvd_inference_diffusion_steps
        n_cond = world_model_conditioning.shape[1]
        noise_5d = noise.reshape(B, T, *self.embed_cwh).permute(0, 2, 1, 3, 4)
        cond_5d = world_model_conditioning.reshape(B, n_cond, *self.embed_cwh).permute(0, 2, 1, 3, 4)
        gen_5d = self.sample(cond_5d, noise_5d.shape, n_steps=n_steps, device=noise.device, use_ema=use_ema, cfg_scale=cfg_scale or 0.0)
        generated = gen_5d.permute(0, 2, 1, 3, 4).reshape(B, T, -1)
        return generated

    def ema_update(self):
        update_ema_model(self.regular_model_submodules, self.ema_model_submodules, self.hparams.ema_model)
