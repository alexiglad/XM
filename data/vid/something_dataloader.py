import cv2
import os, shutil
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import subprocess
import json
import pandas
from data.vid.data_preprocessor import remove_padding, crop_to_square, get_all_frames, handle_corrupt_file
from data.vid.preencode_feature_extractor import hdf5_split_finished
import static_ffmpeg

class SomethingDataset(Dataset):
    def __init__(self, hparams, split, transform, labels_dir = 'labels/', ignore_filepath=""):
        """
        Args:
            dataset_dir (str): Directory containing class folders.
            transform (callable, optional): Optional transform to apply to frames.
            num_frames (int): Number of frames to return from each video.
        """
        self.current_user = os.getenv("USER")
        self.hparams = hparams
        assert self.hparams.dataset_dir != "", "pass --dataset_dir for ssv2 (e.g. $SSV2_DIR)"
        self.dataset_dir = self.hparams.dataset_dir
        self.labels_dir = self.dataset_dir + '/' + labels_dir
        self.transform = transform
        self.num_frames = self.hparams.context_length
        self.time_between_frames = self.hparams.time_between_frames
        self.sampling_rate = self.hparams.sampling_rate
        self.split = split

        if self.split not in ('train','test', 'val'):
            raise(Exception("Split must be train, test, or val"))

        self.hdf5_file = None
        self.frame_lookup = None
        self.class_names = []
        self.file_list = []
        self.labels = []
        self.ignore_filepath = ""
        self.potential_ignore_filepath = ""

        # if this split is marked finished in the preencoded hdf5, skip raw labels/ffprobe entirely
        self.use_hdf5 = self.hparams.preencode_dataset and hdf5_split_finished(self.hparams.hdf5_file_path, self.split)
        if self.use_hdf5:
            with h5py.File(self.hparams.hdf5_file_path, "r") as f:
                raw = f[self.split]['_meta_labels'][()]
                if isinstance(raw, bytes):
                    raw = raw.decode('utf-8')
                self.labels = json.loads(raw)
                self.class_names = json.loads(f[self.split].attrs['class_names'])
                self.file_list = [None] * int(f[self.split].attrs['num_samples'])
            return

        # raw-data path (first preencode run, or no preencode at all)
        self.potential_ignore_filepath = '/'.join([os.getcwd(), "data/vid/corrupt_files/ssv2.txt"])
        if ignore_filepath:
            self.ignore_filepath = self.labels_dir + ignore_filepath
        elif os.path.exists(self.potential_ignore_filepath):
            self.ignore_filepath = self.potential_ignore_filepath
        else:
            self.ignore_filepath = ignore_filepath

        # FFPROBE SETUP
        static_ffmpeg.add_paths()
        self.ffprobe = shutil.which("ffprobe") or os.getenv("FFPROBE_PATH") or hparams.ffprobe_path
        assert self.ffprobe and os.path.exists(self.ffprobe), "ffprobe not found: install ffmpeg, set $FFPROBE_PATH, or pass --ffprobe_path"

        annotations = None
        if self.split == 'val':
            annotations = pandas.read_json(f"{self.labels_dir}/validation.json")
        else:
            annotations = pandas.read_json(f"{self.labels_dir}/{self.split}.json")

        label_dict = pandas.read_json(f"{self.labels_dir}/labels.json", typ='Series')

        if self.split == 'test':
            labels = pandas.read_csv((f"{self.labels_dir}/test-answers.csv"), header=None, sep=';')
            labels.columns = ['id', 'label']
            labels.set_index('id')
            self.labels = list(label_dict[labels.label])
        else:
            self.labels = list(label_dict[annotations.apply(lambda r: r.template.replace("[","").replace("]",""), axis=1)])
        self.class_names = list(label_dict.values)

        self.file_list = list(annotations.apply(lambda r:
        f"{self.dataset_dir}/20bn-something-something-v2/{r.id}.webm", axis=1))

        # self.video_ids = list(annotations.apply(lambda r: f"{r.id}", axis=1))

        # print("Filtering out non-existent video paths for SSV2")
        # self.file_list = list(filter(os.path.exists, self.file_list))

        if self.ignore_filepath:
            with open(self.ignore_filepath) as f:
                self.ignore_files = f.read()
            new_file_list = []
            for fp in self.file_list:
                fp_components = os.path.normpath(fp).split('/')
                new_fp = '/'.join(fp_components[-2:])
                if new_fp not in self.ignore_files:
                    new_file_list.append(fp)
            self.file_list = new_file_list



    def get_length(self, filename):
        command = self.ffprobe + ' -v quiet -print_format json -show_format "{}"'.format(filename)
        data = json.loads(subprocess.check_output(command, shell=True))
        return float(data['format']['duration'])


    def get_num_frames(self, filepath):
        cap = cv2.VideoCapture(filepath)
        estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        count=0
        for i in range(estimate):
            ret, img = cap.read()
            if not ret: break
            count += 1
        
        cap.release()
        return count
    

    def get_num_frames_static(self, filepath):
        if self.frame_lookup == None:
            self.frame_lookup = json.load(open(self.labels_dir+"lengths.json"))
        
        return self.frame_lookup[filepath]
        

    def __len__(self):
        if self.hparams.debug_mode:
            return 1000
        else:
            return len(self.file_list)
        

    def _init_hdf5(self):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hparams.hdf5_file_path, "r")


    def __del__(self):
        if hasattr(self, 'hdf5_file') and self.hdf5_file:
            self.hdf5_file.close()


    def __getitem__(self, idx, _retries=0):
        video_path = self.file_list[idx]
        # video_id   = self.video_ids[idx]
        label      = self.labels[idx]
        try:

            if self.use_hdf5:
                # read straight from HDF-5, no OpenCV
                self._init_hdf5()
                key = f"{self.split}/{idx}"
                # frames = [torch.tensor(self.hdf5_file[key][str(i.item())][:]) for i in frame_idx]
                ds = self.hdf5_file[key]             # h5py.Dataset, shape = (T, …)
                total_frames = ds.shape[0] - 1
                video_len = 1 / 12 * total_frames # use raw framerate of SSV2 which is 12 FPS

                frame_idx = self.get_frame_indices(video_len, total_frames)
                # frames = torch.as_tensor(ds[[int(i) for i in frame_idx]])
                if torch.unique_consecutive(frame_idx).numel() == frame_idx.numel():
                    frames = torch.as_tensor(ds[frame_idx.tolist()])       # one fancy read
                else:
                    # 2) Duplicates present → fall back to per-frame reads to avoid error about increasing list indexing elements
                    frames = torch.stack([torch.as_tensor(ds[int(i)]) for i in frame_idx])
            else:
                video_len    = self.get_length(video_path)
                total_frames = self.get_num_frames(video_path) - 1

                if self.hparams.preencode_dataset: # in this branch use_hdf5 is False, so this split is not finished → we're preencoding it and need all frames
                    frame_idx = torch.arange(total_frames).long()            # all frames
                else:                                                        # training / eval with no preencoding
                    frame_idx = self.get_frame_indices(video_len, total_frames)

                # read from video file
                frames = self.capture_video_frames(video_path, len(frame_idx), frame_idx)
                frames = torch.stack(frames)

            return {"frames": frames, "labels": label, "index": idx}

        except Exception as e:
            if _retries < 10:
                new_idx = random.randint(0, len(self) - 1)
                print(f"Warning: Error loading idx={idx} ({video_path}): {e}. Retrying with idx={new_idx}.")
                return self.__getitem__(new_idx, _retries=_retries + 1)
            log_path = self.ignore_filepath or self.potential_ignore_filepath
            handle_corrupt_file(e, video_path, log_path)


    def capture_video_frames(self, video_path, num_frames, frame_idx_list):
        cap = cv2.VideoCapture(video_path)
        frames = []
        for i in range(num_frames):           
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_list[i].item())
            ret, frame = cap.read()
            if not ret: 
                print(f"frame is blank at {frame_idx_list[i].item()}")

            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_width, original_height = original_frame.shape[:2]
            pil_image = Image.fromarray(original_frame)
                
            if self.hparams.preprocess_data:
                raise NotImplementedError("this is not supported as of now")
 
            if self.transform:
                transformed_frame = self.transform(pil_image)
                frames.append(transformed_frame)
            else:
                raise ValueError("Please specify transform")

        cap.release()

        return frames

            

    def get_frame_indices(self, video_length, total_frames):
        if total_frames < self.num_frames:
            frame_idx_list = torch.linspace(0, total_frames, steps=self.num_frames).long() # just duplicate frames
        elif self.hparams.use_raw_framerate:
            start_frame = 0 if self.hparams.no_randomness_dataloader else int(random.random() * (total_frames-self.num_frames))
            frame_idx_list = torch.arange(start_frame, start_frame+self.num_frames).long()
        elif self.hparams.sampling_rate != 0:
            required_frames = self.hparams.sampling_rate * (self.num_frames-1)
            if required_frames > total_frames:
                frame_idx_list = torch.linspace(0, total_frames, steps=self.num_frames).long() # Default case when you can't put the desired space between frames
            else:
                start_frame = 0 if self.hparams.no_randomness_dataloader else int(random.random() * (total_frames-required_frames))
                frame_idx_list = torch.linspace(start_frame, start_frame+required_frames, steps=self.num_frames).long()
        else: # uses time_between_frames hparam
            required_length = self.time_between_frames * (self.num_frames-1)
            if required_length > video_length:
                frame_idx_list = torch.linspace(0, total_frames, steps=self.num_frames).long() # Default case when you can't put the desired space between frames
            else:
                start_time = 0 if self.hparams.no_randomness_dataloader else random.random() * (video_length-required_length)
                end_time = start_time + required_length

                frame_idx_list = (total_frames/video_length
                                *torch.linspace(start_time, end_time, steps=self.num_frames)).long()
                                
        return frame_idx_list
