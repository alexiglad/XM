import torch
import os
import json
import h5py
import torch
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from model.model_utils import *

# from model.vid.vjepa.model_utils.utils import init_model
# from utils.vjepa_transforms.utils import ClipAggregation


def hdf5_split_finished(hdf5_file_path, split):
	"""Only True once preencoding fully completed and wrote the 'finished' attr on the split group. Used by SomethingDataset and the preencode trigger in base_model_trainer to avoid the old 1GB-size heuristic, which is unreliable for partial runs or small datasets."""
	if not os.path.exists(hdf5_file_path):
		return False
	try:
		with h5py.File(hdf5_file_path, "r") as f:
			return split in f and bool(f[split].attrs.get('finished', False))
	except OSError:
		return False


def collate_fn_pad(batch):
		frames     = [sample['frames'] for sample in batch]   # list of [T, C, H, W]
		video_ids  = [sample['index']  for sample in batch]

		# Pad sequences along time dimension
		lengths = [v.shape[0] for v in frames]
		padded_videos = pad_sequence(frames, batch_first=True)  # shape: [B, max_T, C, H, W]

		return padded_videos, lengths, list(video_ids)

def preencode_dataset_features(hparams, encoder, dataset, hdf5_file_path, split):
	print(f"Encoding {len(dataset)} videos in {split} split to store at {hdf5_file_path}")

	device = "cuda" if torch.cuda.is_available() else "cpu"

	# configured for tdv models
	encoder.to(device)
	encoder.eval()

	# freeze encoder params
	for p in encoder.parameters():
		p.requires_grad = False

	dataloader = DataLoader(dataset, batch_size=hparams.batch_size_per_device, shuffle=False, pin_memory=True, drop_last=False, num_workers=hparams.num_workers_per_device, collate_fn=collate_fn_pad)

	# write directly to the final path; per-split 'finished' attr (set only after the loop) is the atomicity guard, so a partially-written split is ignored by hdf5_split_finished on the next run
	with h5py.File(hdf5_file_path, "a") as h5f:
		split_group = h5f.require_group(split)

		# write split-level metadata so SomethingDataset can init from hdf5 without raw videos/labels
		if '_meta_labels' in split_group:
			del split_group['_meta_labels']
		split_group.create_dataset('_meta_labels', data=json.dumps(list(dataset.labels), default=str))
		split_group.attrs['class_names'] = json.dumps(list(dataset.class_names), default=str)
		split_group.attrs['num_samples'] = len(dataset)

		for batch in tqdm(dataloader, desc=f"Preencoding {split} dataset features: "):
			frames, lengths, video_ids = batch
			batch_size, max_num_frames, c, h, w = frames.shape

			with torch.no_grad():
				frames = frames.to(device)
				if hparams.backbone_type == "dinov2":
					encoded_frames_output = encoder(frames.reshape(-1, c, h, w), is_training=True)

					encoded_class_tokens = encoded_frames_output['x_norm_clstoken']																	# [batch_size * num_frames, dim]
					encoded_patch_tokens = encoded_frames_output['x_norm_patchtokens'] 																# [batch_size * num_frames, num_patches, dim]

					encoded_class_tokens = encoded_class_tokens.reshape(batch_size, max_num_frames, -1)
					encoded_patch_tokens = encoded_patch_tokens.reshape(batch_size, max_num_frames, -1, encoded_class_tokens.shape[-1])

					encodings = torch.cat((encoded_class_tokens.unsqueeze(2), encoded_patch_tokens), dim=2)
				elif hparams.backbone_type == "mae":
					num_patches = (h//16) * (w//16)						
					noise = torch.zeros((batch_size*max_num_frames, num_patches), dtype=torch.float32)

					# -- mae returns output in [seq_len, num_patches, dim]
					encodings = encoder(frames.reshape(-1, c, h, w), noise=noise, return_dict=True).last_hidden_state					
					encodings = encodings.reshape(batch_size, max_num_frames, -1, encodings.shape[-1])												# [batch_size, num_frames, num_patches, dim]
				elif hparams.backbone_type == "vae":
					frames = frames.reshape(batch_size*max_num_frames, c, w, h) # B*S, C, W, H
					encodings = get_encoded_images(frames, hparams.backbone_type, encoder, sdxl_vae_standardization = True)
					encodings = encodings.reshape(batch_size, max_num_frames, *encodings.shape[1:]) # B, S, C, W, H
				elif hparams.backbone_type == "rae":
					frames = frames.reshape(batch_size*max_num_frames, c, w, h) # B*S, C, W, H
					encodings = get_encoded_images(frames, hparams.backbone_type, encoder)
					encodings = encodings.reshape(batch_size, max_num_frames, *encodings.shape[1:]) # B, S, C, W, H
				else:
					raise NotImplementedError(f"have not yet implemented backbone_type: {hparams.backbone_type}")

			# store features in hdf5 file
			encodings = encodings.cpu().numpy()

			for i in range(batch_size):
				video_id = str(video_ids[i])

                # --- SAFETY CHECK: Delete if exists, so we don't crash on restart ---
				if video_id in split_group:
					del split_group[video_id]

				split_group.create_dataset(video_id, data=encodings[i, :lengths[i]])

		missing = [str(i) for i in range(len(dataset)) if str(i) not in split_group] # a failed __getitem__ retries with a random substitute index, so its own key never gets written
		assert not missing, f"preencode incomplete: {len(missing)} videos failed to encode (e.g. indices {missing[:5]}), fix or remove them and rerun"
		split_group.attrs['finished'] = True # set only after the loop completes so partial runs don't get trusted by hdf5_split_finished

	print(f"Encoded {len(dataset)} videos in {split} split and stored in {hdf5_file_path}")
