import gc
import itertools
import math
import shutil
import sys
import os
import traceback
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

_METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'common_metrics_on_video_quality')
if _METRICS_DIR not in sys.path:
    sys.path.insert(0, _METRICS_DIR)
from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained

from datetime import timedelta

from model.model_utils import reset_image_encoder_decoder, get_encoded_images, denormalize, temporary_ddp_timeout
from data.vid.something_dataloader import SomethingDataset

NUM_VIDEOS_TO_SAVE = 10


class VideoFVDCallback(pl.Callback):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.reset_image_encoder_decoder = False  # prevent bug with sdxl vae


    def on_fit_start(self, trainer, pl_module):
        """
        Called at the start of training.
        """
        # uncomment this for debugging where dont have to wait for lots of training
        # print(f"running online FVD evaluation at epoch {trainer.current_epoch}")
        # self.run_online_fvd_evaluation(trainer, pl_module)
        pass


    def on_train_epoch_end(self, trainer, pl_module):
        hparams = pl_module.hparams
        if trainer.current_epoch % hparams.run_online_fvd_eval_every_n_epochs == 0:
            print(f"running online FVD evaluation at epoch {trainer.current_epoch}")
            with temporary_ddp_timeout(timedelta(minutes=120)): # rank 0 runs FVD (minutes) while other ranks wait at the trailing barrier
                if trainer.global_rank == 0:
                    self.run_online_fvd_evaluation(trainer, pl_module)
                trainer.strategy.barrier()

    @torch.no_grad()
    def run_online_fvd_evaluation(self, trainer, pl_module):
        hparams = pl_module.hparams
        if not self.reset_image_encoder_decoder:
            reset_image_encoder_decoder(self.hparams, pl_module.model)
            self.reset_image_encoder_decoder = True

        device = pl_module.device
        model = pl_module.model
        model_was_training = model.training
        model.eval()

        num_videos = hparams.online_fvd_eval_num_videos

        if hparams.backbone_type == "rae":
            raise NotImplementedError("have not yet implemented online FVD eval for RAE")

        use_ema = hparams.ema_model != 1.0 # uses EMA if model is being EMA'ed

        h, w = hparams.image_dims[0], hparams.image_dims[1]
        embed_cwh = (4, h // 8, w // 8)

        cfg_scales = hparams.online_fvd_cfg_scales

        log_dir = Path(f"./logs/eval/online_fvd_{hparams.run_name}")
        if log_dir.exists():
            shutil.rmtree(log_dir)

        # build val dataloader
        if hparams.dataset_name in ('something', 'smth'):
            if hparams.vae_normalization:
                transform = transforms.Compose([
                    transforms.Resize((h, w)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((h, w)),
                    transforms.ToTensor(),
                ])
            val_ds = SomethingDataset(hparams, split='val', transform=transform)
        else:
            raise NotImplementedError(f"online FVD eval not yet implemented for dataset '{hparams.dataset_name}'")

        if num_videos > len(val_ds):
            print(f"Warning: online_fvd_eval_num_videos ({num_videos}) > val set size ({len(val_ds)}); "
                  f"FVD will be computed on {len(val_ds)} videos instead.")
            num_videos = len(val_ds)

        # NOTE the vae decode below is fed max_bs * T latents in one call, not max_bs -- measured peak for
        # max_bs=64, T=10 at 128x128 (sd-vae-ft-ema) is ~44 GiB, on top of whatever training still has resident
        # (empty_cache only runs in the finally). lower this divisor or chunk the decode if the eval ooms.
        max_bs = math.ceil(hparams.batch_size_per_device / 4)  # prevent gpu ooms due to vae taking tons of space
        loader = DataLoader(val_ds, batch_size=max_bs,
                            shuffle=False, num_workers=4, drop_last=True)

        assert hparams.video_task == "gc_world_model", "online fvd only supports goal conditioned world model as a task for now"

        idx = torch.as_tensor(hparams.world_modeling_given_frames, device=device)
        num_batches = math.ceil(num_videos / max_bs)
        n_steps = hparams.fvd_inference_diffusion_steps

        i3d = load_i3d_pretrained(device=device)

        try:
            for cfg_scale in cfg_scales:
                try:
                    gen_dir = log_dir / f"gen_cfg{cfg_scale}"
                    gen_dir.mkdir(parents=True, exist_ok=True)

                    feats_real = np.empty((0, 400))
                    feats_gen  = np.empty((0, 400))
                    collected = 0

                    for batch in tqdm(itertools.islice(loader, num_batches), total=num_batches,
                                      desc=f"FVD eval epoch {trainer.current_epoch} cfg={cfg_scale}"):
                        frames = batch['frames'].to(device)  # (B, T, C or 4, H, W)
                        B, T = frames.shape[:2]

                        # get latents (B, T, D)
                        if hparams.preencode_dataset:
                            latents = frames.reshape(B, T, -1)  # already encoded and sdxl-scaled
                        else:
                            flat = frames.reshape(B * T, *frames.shape[2:])
                            latents = get_encoded_images(
                                flat, hparams.backbone_type, model.image_encoder,
                                sdxl_vae_standardization=hparams.sdxl_vae_standardization
                            ).reshape(B, T, -1)

                        # world model conditioning from GT frames
                        world_model_conditioning = latents[:, idx].clone()

                        # generate
                        noise = torch.randn_like(latents)
                        generated = model.generate_samples(noise, world_model_conditioning, use_ema=use_ema, cfg_scale=cfg_scale, n_steps=n_steps)

                        if not hparams.gc_world_model_unconditional:
                            generated[:, idx] = world_model_conditioning

                        # decode generated → [0, 1]
                        gen_pixels = model.image_encoder.decode(
                            generated.reshape(-1, *embed_cwh) / 0.18215
                        ).sample
                        gen_pixels = denormalize(
                            gen_pixels.unsqueeze(0), hparams.dataset_name, device,
                            hparams.custom_image_normalization, vae_normalization=True
                        ).squeeze(0).reshape(B, T, 3, h, w)

                        # real pixels: use original frames directly (skip expensive VAE decode round-trip)
                        if hparams.preencode_dataset:
                            real_pixels = model.image_encoder.decode(
                                latents.reshape(-1, *embed_cwh) / 0.18215
                            ).sample
                            real_pixels = denormalize(
                                real_pixels.unsqueeze(0), hparams.dataset_name, device,
                                hparams.custom_image_normalization, vae_normalization=True
                            ).squeeze(0).reshape(B, T, 3, h, w)
                        else:
                            real_pixels = frames * 0.5 + 0.5 if hparams.vae_normalization else frames

                        # save gen videos to disk for visual inspection
                        for b in range(min(B, max(0, NUM_VIDEOS_TO_SAVE - collected))):
                            vid_idx = collected + b
                            for t in range(T):
                                save_image(gen_pixels[b, t].clamp(0, 1), str(gen_dir / f"{vid_idx:04d}_t{t:02d}.png"))

                        # extract i3d features — BCTHW format
                        feats_real = np.vstack([feats_real, get_fvd_feats(real_pixels.permute(0, 2, 1, 3, 4), i3d=i3d, device=device)])
                        feats_gen  = np.vstack([feats_gen,  get_fvd_feats(gen_pixels.permute(0, 2, 1, 3, 4),  i3d=i3d, device=device)])
                        collected += B

                    print(f"Computing FVD over {collected} videos (cfg={cfg_scale})...")
                    fvd = frechet_distance(feats_gen[:num_videos], feats_real[:num_videos])

                    trainer.logger.experiment.log({f"online_eval/fvd_cfg={cfg_scale}": fvd, "trainer/global_step": trainer.global_step})
                    print(f"Online FVD epoch={trainer.current_epoch} cfg={cfg_scale}: {fvd:.2f}")

                except Exception:
                    # per-cfg so one failure (usually an oom in the decode) doesn't cost us the other cfg scales
                    print(f"Error during online FVD evaluation at cfg={cfg_scale}, skipping this cfg scale:")
                    traceback.print_exc()
                    torch.cuda.empty_cache()  # drop whatever the failed cfg left allocated before trying the next

        except Exception:
            # never propagate: only rank 0 runs this and every rank still has to reach the trailing barrier in
            # on_train_epoch_end, so raising here would hang the job instead of just losing the metric
            print("Error during online FVD evaluation:")
            traceback.print_exc()

        finally:
            if model_was_training:
                model.train()
            torch.cuda.empty_cache()
            gc.collect()
