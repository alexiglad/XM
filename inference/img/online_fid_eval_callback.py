import gc
import os
import shutil
from datetime import timedelta

import torch
import torch.distributed as dist
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, Dataset
import wandb
from pathlib import Path
from tqdm import tqdm
import json
import torch_fidelity

from model.model_utils import *
from data.img.imagenet_dataloader import ImageNetDataset


class ImageNetValForFidelity(Dataset):
    """Wraps ImageNetDataset for torch_fidelity (must return a bare uint8 tensor)."""
    def __init__(self, hparams):
        image_size = hparams.image_dims[0]
        self.ds = ImageNetDataset(hparams, split="validation", transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
        ]))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]['image']  # uint8 (3, H, W)


def _fid_stats_path(image_size: int) -> str:
    if image_size == 256:
        return "inference/img/fid_stats/adm_in256_stats.npz"
    elif image_size == 64:
        return "inference/img/fid_stats/adm_in64_stats.npz"
    raise NotImplementedError("Only image_size=256 and 64 are supported for torch-fidelity FID stats")


# Class balance: global sample pool is exactly class-balanced. Per-rank labels are a contiguous
# slice of labels_all; fine because FID is computed by rank 0 over the union of samples in samples_dir.
class ImageFIDCallback(pl.Callback):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.reset_image_encoder_decoder = False # prevent a bug with sdxl vae
        self.val_dataset = None


    def on_fit_start(self, trainer, pl_module):
        """
        Called at the start of training.
        """
        # uncomment this for debugging where dont have to wait for lots of training
        # print(f"running online FID evaluation at epoch {trainer.current_epoch}")
        # self.run_online_fid_evaluation(trainer, pl_module, "valid")
        pass
    

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called after every train epoch.
        """
        hparams = pl_module.hparams
        
        if ((trainer.current_epoch % hparams.run_online_fid_eval_every_n_epochs)) == 0:
            print(f"running online FID evaluation at epoch {trainer.current_epoch}")
            self.run_online_fid_evaluation(trainer, pl_module, "valid")
        

    def run_online_fid_evaluation(self, trainer, pl_module, phase):
        with temporary_ddp_timeout(timedelta(minutes=120)): # rank 0 runs torch_fidelity (minutes) while other ranks wait at the trailing barrier
            self._run_online_fid_evaluation(trainer, pl_module, phase)

    def _run_online_fid_evaluation(self, trainer, pl_module, phase):
        hparams = pl_module.hparams

        if not self.reset_image_encoder_decoder: # this is done to prevent bug where loading ckpt image encoder doesnt work well, not sure why ckpt image decoder doesnt load well, maybe related to HF or PL
            reset_image_encoder_decoder(self.hparams, pl_module.model)
            self.reset_image_encoder_decoder = True

        model_was_training = pl_module.model.training
        pl_module.model.eval()

        # Parameters
        num_samples = hparams.online_fid_eval_num_samples
        batch_size = hparams.batch_size_per_device
        image_size = hparams.image_dims[0]
        cfg_scales = hparams.online_fid_cfg_scales
        use_ema = hparams.ema_model != 1.0 # uses EMA if model is being EMA'ed

        log_dir = Path("./logs/eval")
        log_dir = log_dir / f"online_fid_epoch_{trainer.current_epoch}_{hparams.run_name}"

        if trainer.global_rank == 0:
            if log_dir.exists():
                shutil.rmtree(log_dir, ignore_errors=True)

        trainer.strategy.barrier()

        all_metrics = {}
        for cfg_scale in cfg_scales:
            samples_dir = log_dir / f"samples_cfg{cfg_scale}"

            if trainer.global_rank == 0:
                samples_dir.mkdir(parents=True, exist_ok=True)

            trainer.strategy.barrier()

            self.generate_images_distributed(
                pl_module.model,
                num_samples=num_samples,
                batch_size=batch_size,
                cfg_scale=cfg_scale,
                image_size=image_size,
                device=pl_module.device,
                use_ema=use_ema,
                out_dir=samples_dir,
                rank=trainer.global_rank,
                world_size=trainer.world_size
            )

            trainer.strategy.barrier()

            if trainer.global_rank == 0:
                print(f"Computing FID and other metrics (cfg_scale={cfg_scale})...")
                try:
                    metrics = self.compute_metrics_with_torch_fidelity(samples_dir, image_size, pl_module.device)

                    all_metrics[f"online_eval/fid_cfg={cfg_scale}"] = metrics["frechet_inception_distance"]
                    all_metrics[f"online_eval/inception_score_mean_cfg={cfg_scale}"] = metrics["inception_score_mean"]
                    all_metrics[f"online_eval/precision_cfg={cfg_scale}"] = metrics["precision"]
                    all_metrics[f"online_eval/recall_cfg={cfg_scale}"] = metrics["recall"]

                    print(f"Online FID (cfg_scale={cfg_scale}): {metrics['frechet_inception_distance']}")
                except Exception as e:
                    print(f"Error computing online FID (cfg_scale={cfg_scale}): {e}")

            trainer.strategy.barrier()

        if trainer.global_rank == 0:
            if all_metrics:
                trainer.logger.experiment.log({**all_metrics, "trainer/global_step": trainer.global_step})

        if trainer.global_rank == 0:
            if log_dir.exists():
                shutil.rmtree(log_dir, ignore_errors=True)

        trainer.strategy.barrier()

        torch.cuda.empty_cache()
        gc.collect()

        if model_was_training:
            pl_module.model.train()

    @torch.no_grad()
    def generate_images_distributed(self, model, num_samples: int, batch_size: int, cfg_scale: float, image_size: int, device: torch.device, use_ema: bool, out_dir: Path, rank: int = 0, world_size: int = 1):
        """
        Each rank will generate a disjoint subset of the total num_samples.
        Filenames are offset so no collisions occur.
        """
        if model.hparams.backbone_type == "rae":
            C, H, W = 768, 16, 16
        else:
            C, H, W = 4, image_size // 8, image_size // 8
        num_classes = model.hparams.num_classes

        # Determine per-rank assignment (fair split)
        per_rank = num_samples // world_size
        rem = num_samples % world_size
        # First `rem` ranks get one extra
        my_num = per_rank + (1 if rank < rem else 0)

        if my_num == 0:
            return  # nothing for this rank

        # To ensure class-balancing as before but split across ranks,
        # construct the global labels_all sequence deterministically, then slice by global offsets.
        # Build labels_all for entire dataset (on each rank) but slice the range this rank is responsible for.
        base = num_samples // num_classes
        extra = num_samples % num_classes
        labels_all = torch.arange(num_classes, device=device).repeat_interleave(base)
        if extra > 0:
            labels_all = torch.cat([labels_all, torch.arange(extra, device=device)])
        labels_all = labels_all.long()

        # compute the global start index for this rank
        start_idx = (per_rank * rank) + min(rank, rem)
        end_idx = start_idx + my_num
        labels = labels_all[start_idx:end_idx]

        # Starting file index offset for this rank to avoid filename collisions
        global_start_file_idx = start_idx

        n = 0
        idx = global_start_file_idx
        # Only show pbar on rank 0
        pbar = tqdm(total=my_num, desc=f"Generating images (rank {rank})", disable=(rank != 0))
        with torch.amp.autocast('cuda', enabled=False): # force fp32 for inference precision
            while n < my_num:
                b = min(batch_size, my_num - n)
                labels_b = labels[n:n + b].to(device)
                shape = (b, C, H, W)
                z = torch.randn(shape, device=device)

                samples = model.generate_samples(z, labels_b, cfg_scale=cfg_scale, use_ema=use_ema)

                if model.hparams.backbone_type == "rae":
                    imgs = model.image_encoder.decode(samples).clamp(0, 1)
                else: # vae
                    imgs = model.image_encoder.decode(samples / 0.18215).sample
                    imgs = (imgs + 1) / 2
                    imgs = imgs.clamp(0, 1)

                for i in range(b):
                    # filename: zero-padded global index
                    save_image(imgs[i], str(out_dir / f"{idx:08d}.png"))
                    idx += 1
                n += b
                pbar.update(b)
        pbar.close()

    @torch.no_grad()
    def compute_metrics_with_torch_fidelity(self, samples_dir: Path, image_size: int, device: torch.device) -> dict:
        stats_path = _fid_stats_path(image_size)
        use_cuda = (getattr(device, "type", str(device)).startswith("cuda") and torch.cuda.is_available())

        # split into two calls: torch_fidelity silently ignores fid_statistics_file when input2 is provided,
        # so FID/ISC go through the precomputed ADM stats path (comparable to prior runs) and PRC gets its own call
        fid_isc = torch_fidelity.calculate_metrics(
            input1=str(samples_dir),
            input2=None,
            fid_statistics_file=stats_path,
            cuda=use_cuda,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )

        # PRC is best effort: it loads the val set and needs its own feature pass, so don't let it lose FID/ISC above
        prc = {}
        try:
            if self.val_dataset is None:
                print("Loading ImageNet validation set for precision/recall...")
                self.val_dataset = ImageNetValForFidelity(self.hparams)

            # cache the val features, otherwise they are re-extracted for every cfg scale of every eval. cache_root sits
            # beside the data it came from so a different dataset_dir gets its own cache; n= guards partial downloads
            hf_home = os.getenv('HF_HOME')
            cache_root = f"{self.hparams.dataset_dir if self.hparams.dataset_dir != '' else hf_home}/fidelity_cache"
            prc = torch_fidelity.calculate_metrics(
                input1=str(samples_dir),
                input2=self.val_dataset,
                input2_cache_name=f"imagenet_val_{image_size}_n{len(self.val_dataset)}_prc",
                cache_root=cache_root,
                cuda=use_cuda,
                isc=False,
                fid=False,
                kid=False,
                prc=True,
                verbose=False,
            )
        except Exception as e:
            print(f"Error computing precision/recall (FID/ISC unaffected): {e}")
        out = {
            "frechet_inception_distance": float(fid_isc.get("frechet_inception_distance", float("nan"))),
            "inception_score_mean": float(fid_isc.get("inception_score_mean", float("nan"))),
            "inception_score_std": float(fid_isc.get("inception_score_std", float("nan"))),
            "precision": float(prc.get("precision", float("nan"))),
            "recall": float(prc.get("recall", float("nan"))),
        }
        return out
