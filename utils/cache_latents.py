import json
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file
import bisect


def cache_latents(dataset, encoder, cache_file, backbone_type, batch_size=32, num_workers=12, device='cuda', shard_size: int = 0):
    """Cache encoded latents to disk in sharded safetensors.

    shard_size: max samples per shard file. 0 = auto-calculate to target ~2GB per shard.
    """
    from model.model_utils import get_encoded_images

    cache_file = Path(cache_file)
    parts_dir = cache_file.with_suffix(cache_file.suffix + ".parts")
    if cache_file.exists() or parts_dir.exists():
        print(f"Cache exists: {cache_file}")
        return str(cache_file)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    encoder.eval()
    encoder = encoder.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                       shuffle=False, pin_memory=True, drop_last=False)

    # Auto-calculate shard_size from first batch if not specified
    if shard_size <= 0:
        with torch.no_grad():
            sample_batch = next(iter(loader))
            sample_latent = get_encoded_images(sample_batch['image'][:1].to(device), backbone_type, encoder)
            bytes_per_sample = sample_latent.numel() * sample_latent.element_size()
            target_shard_bytes = 2 * 1024**3  # 2 GB per shard
            shard_size = max(1000, target_shard_bytes // bytes_per_sample)
        # Re-create loader since we consumed from iterator
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                           shuffle=False, pin_memory=True, drop_last=False)
        print(f"[cache_latents] Auto shard_size={shard_size} ({bytes_per_sample} bytes/sample, targeting ~2GB shards)")

    buffer_latents = []
    buffer_labels = []
    buffer_count = 0
    part_sizes = []
    part_files = []
    part_idx = 0

    def flush_part():
        nonlocal buffer_latents, buffer_labels, buffer_count, part_idx
        if buffer_count == 0:
            return
        latents_cat = torch.cat(buffer_latents, dim=0).contiguous()
        labels_cat = torch.cat(buffer_labels, dim=0).contiguous()
        part_path = parts_dir / f"part_{part_idx:05d}.safetensors"
        save_file({'latents': latents_cat, 'labels': labels_cat}, part_path)
        part_sizes.append(int(labels_cat.shape[0]))
        part_files.append(str(part_path.name))
        buffer_latents = []
        buffer_labels = []
        buffer_count = 0
        part_idx += 1

    with torch.no_grad():
        for batch in tqdm(loader, desc="Caching"):
            images = batch['image'].to(device)
            latents = get_encoded_images(images, backbone_type, encoder).cpu()
            labels = batch['label']
            buffer_latents.append(latents)
            buffer_labels.append(labels)
            buffer_count += latents.shape[0]
            if buffer_count >= shard_size:
                flush_part()

    flush_part()

    manifest = {
        'parts_dir': parts_dir.name,
        'parts': [{'file': f, 'size': s} for f, s in zip(part_files, part_sizes)]
    }
    (parts_dir / 'manifest.json').write_text(json.dumps(manifest))

    # create a small sentinel file so existing existence checks pass
    cache_file.write_text('sharded')

    total = sum(part_sizes)
    print(f"Cached {total} samples to {parts_dir}")
    return str(cache_file)


class CachedLatentsDataset(torch.utils.data.Dataset):
    def __init__(self, cache_file):
        self.cache_file = Path(cache_file)
        parts_dir = self.cache_file.with_suffix(self.cache_file.suffix + ".parts")
        if parts_dir.exists():
            self._sharded = True
            manifest = json.loads((parts_dir / 'manifest.json').read_text())
            self._parts_dir = parts_dir
            self._parts = manifest['parts']
            self._sizes = [p['size'] for p in self._parts]
            self._cumsum = []
            total = 0
            for s in self._sizes:
                total += s
                self._cumsum.append(total)
            self._current_part_idx = None
            self._current_tensors = None
        else:
            self._sharded = False
            tensors = load_file(str(self.cache_file))
            self.latents = tensors['latents']
            self.labels = tensors['labels']

    def __len__(self):
        if self._sharded:
            return 0 if not self._cumsum else self._cumsum[-1]
        return len(self.latents)

    def _load_part(self, part_idx: int):
        if self._current_part_idx == part_idx and self._current_tensors is not None:
            return
        part_path = self._parts_dir / self._parts[part_idx]['file']
        tensors = load_file(str(part_path))
        self._current_part_idx = part_idx
        self._current_tensors = tensors

    def __getitem__(self, idx):
        if not self._sharded:
            return {
                'latent': self.latents[idx],
                'label': int(self.labels[idx])
            }
        part_idx = bisect.bisect_right(self._cumsum, idx)
        start = 0 if part_idx == 0 else self._cumsum[part_idx - 1]
        local_idx = idx - start
        self._load_part(part_idx)
        latents = self._current_tensors['latents']
        labels = self._current_tensors['labels']
        return {
            'latent': latents[local_idx],
            'label': int(labels[local_idx])
        }


def _is_rank_zero():
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _barrier():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cache_image_dataset(hparams, transform, device):
    from pathlib import Path
    from data.img.imagenet_dataloader import ImageNetDataset
    from model.model_utils import load_image_encoder

    backbone = hparams.backbone_type
    cache_dir = Path(hparams.cached_img_latents_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    res = f"{hparams.image_dims[0]}x{hparams.image_dims[1]}"
    train_cache = cache_dir / f"imagenet_train_{res}_{backbone}.safetensors"
    val_cache = cache_dir / f"imagenet_val_{res}_{backbone}.safetensors"

    def _exists(p: Path):
        parts_dir = p.with_suffix(p.suffix + ".parts")
        return p.exists() or parts_dir.exists()

    need_train = not _exists(train_cache)
    need_val = not _exists(val_cache)

    if _is_rank_zero() and (need_train or need_val):
        encoder = load_image_encoder(backbone, hparams.vit_backbone_size, device=device, use_ema=True)
        if need_train:
            ds = ImageNetDataset(hparams, split='train', transform=transform)
            cache_latents(ds, encoder, str(train_cache), backbone)
        if need_val:
            ds = ImageNetDataset(hparams, split='val', transform=transform)
            cache_latents(ds, encoder, str(val_cache), backbone)
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if need_train or need_val:
        _barrier()
    return str(train_cache), str(val_cache)
