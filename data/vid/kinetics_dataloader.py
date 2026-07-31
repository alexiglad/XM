import cv2
import os, shutil
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import subprocess
import json
import pandas as pd
import getpass
from data.vid.data_preprocessor import remove_padding, crop_to_square, get_all_frames, handle_corrupt_file

class Kinetics400Dataset(Dataset):
    def __init__(self, hparams, split, transform, labels_dir = 'annotations/', ignore_filepath=""):
        """
        Args:
            dataset_dir (str): Directory containing class folders.
            transform (callable, optional): Optional transform to apply to frames.
            num_frames (int): Number of frames to return from each video.
        """
        self.current_user = getpass.getuser()
        self.hparams = hparams
        assert self.hparams.dataset_dir != "", "pass --dataset_dir for k400 (e.g. $K400_DIR)"
        self.dataset_dir = self.hparams.dataset_dir
        self.labels_dir = self.dataset_dir + '/' + labels_dir
        self.transform = transform
        self.num_frames = self.hparams.context_length
        self.time_between_frames = self.hparams.time_between_frames
        self.split = split
        
        self.potential_ignore_filepath = '/'.join([os.getcwd(), "data/vid/corrupt_files/k400.txt"])
        if ignore_filepath:
            self.ignore_filepath = self.labels_dir + ignore_filepath
        elif os.path.exists(self.potential_ignore_filepath):
            self.ignore_filepath = self.potential_ignore_filepath
        else:
            self.ignore_filepath = ignore_filepath
        
        # FFPROBE SETUP
        self.ffprobe = shutil.which("ffprobe") or os.getenv("FFPROBE_PATH") or hparams.ffprobe_path
        assert self.ffprobe and os.path.exists(self.ffprobe), "ffprobe not found: install ffmpeg, set $FFPROBE_PATH, or pass --ffprobe_path"
        
        self.class_names = []
        self.file_list = []
        self.labels = []
        self.frame_lookup = None

        if self.hparams.preprocess_data:
            if not os.path.exists('./logs/debug'): os.mkdir('./logs/debug')
            if not os.path.exists('./logs/debug/images/'): os.mkdir('./logs/debug/images')
        
        
        
        if self.split not in ('train','test', 'val'):
            raise(Exception("Split must be train, test, or val"))
            
        annotations = pd.read_csv(f"{self.labels_dir}{self.split}.csv")
        
        self.labels = list(annotations.label)
        self.class_names = list(set(annotations.label)) #remove duplicates then convert back to list
        
        
        self.file_list = list(annotations.apply(lambda r: 
        f"{self.dataset_dir}/{split}/{r.youtube_id}_{str(r.time_start).zfill(6)}_{str(r.time_end).zfill(6)}.mp4"
                                                .replace("//", "/")                                    
                                                , axis=1))
        
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
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_num_frames_static(self, filepath):
        if self.frame_lookup == None:
            self.frame_lookup = json.load(open(self.labels_dir+"lengths.json"))
        
        return self.frame_lookup[filepath]
        
    def __len__(self):
        if self.hparams.debug_mode:
            return 1000
        else:
            return len(self.file_list)

    def __getitem__(self, idx):
        video_path = self.file_list[idx]
        try:
            video_length = self.get_length(video_path)
            label = self.labels[idx]

            cap = cv2.VideoCapture(video_path)
            total_frames = self.get_num_frames(video_path)-1
            frame_idx_list = []

            if total_frames < self.num_frames:
                frame_idx_list = torch.linspace(0, total_frames, steps=self.num_frames).long() # just duplicate frames
            elif self.hparams.use_raw_framerate:
                if self.hparams.no_randomness_dataloader:
                    start_frame = 0
                else: # normal condition
                    start_frame = random.random() * (total_frames-self.num_frames)
                frame_idx_list = torch.arange(start_frame, start_frame+self.num_frames).long()
            elif self.hparams.sampling_rate != 0:
                required_frames = self.hparams.sampling_rate * (self.num_frames-1)
                if required_frames > total_frames:
                    # print("required_frames was > total_frames", required_frames, total_frames)
                    frame_idx_list = torch.linspace(0, total_frames, steps=self.num_frames).long() # Default case when you can't put the desired space between frames
                else:
                    start_frame = int(random.random() * (total_frames-required_frames))
                    frame_idx_list = torch.linspace(start_frame, start_frame+required_frames, steps=self.num_frames).long()
            else: # uses time_between_frames hparam
                required_length = self.time_between_frames * (self.num_frames-1)
                if required_length > video_length:
                    frame_idx_list = torch.linspace(0, total_frames, steps=self.num_frames).long() # Default case when you can't put the desired space between frames
                else:
                    start_time = random.random() * (video_length-required_length)
                    end_time = start_time + required_length

                    frame_idx_list = (total_frames/video_length
                                    *torch.linspace(start_time, end_time, steps=self.num_frames)).long()


            frames = []


            for i in range(self.num_frames):           
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_list[i].item())
                ret, frame = cap.read()
                if not ret: 
                    continue
                    #print(f"frame is blank at {frame_idx_list[i].item()}")
                    # with open(f'/scratch/{self.current_user}/kinetics-dataset/k400/annotations/error_log.txt', 'a') as f:
                    #     f.write('\n'+video_path)
                    #     f.write(f"\nframe is blank at {frame_idx_list[i].item()}")

                original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                original_width, original_height = original_frame.shape[:2]
                pil_image = Image.fromarray(original_frame)
                
                if self.hparams.preprocess_data:
                    frame, cropped = remove_padding(original_frame)  # Assuming this function is defined
                    if frame.shape[0] < original_height / 2 or frame.shape[1] < original_width / 2:
                        # Call remove_padding again with different threshold values
                        frame, _ = remove_padding(frame, threshold_value=10, threshold_ratio=0.9)
                        if frame.shape[0] < original_height / 2 or frame.shape[1] < original_width / 2:
                            frame, _ = remove_padding(frame, threshold_value=10, threshold_ratio=0.95)
                            #code below is for debugging
                            # if frame.shape[0] < original_height / 2 or frame.shape[1] < original_width / 2:
                            #     save_path = f"./logs/debug/images/special/{video_path}.png"  # Specify the path where you want to save the image
                            #     cv2.imwrite(save_path, cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR))

                    if self.hparams.crop_all_samples or cropped:
                        frame = crop_to_square(frame)  # Assuming this function is defined, we crop all images generally since dont want non uniform distortion

                #code below is for debugging########################################################################
                    # cropped_path = f"./logs/debug/images/{idx}_cropped.png"  # Specify the path where you want to save the image
                    # cv2.imwrite(cropped_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    # uncropped_path = f"./logs/debug/images/{idx}_uncropped.png"  # Specify the path where you want to save the image
                    # cv2.imwrite(uncropped_path, cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR))
                ##########################################################################################
                
                if self.transform:
                    transformed_frame = self.transform(pil_image)
                    frames.append(transformed_frame)
                else:
                    raise ValueError("Please specify transform")


            cap.release()

            frames = torch.stack(frames)
            if self.hparams.model_name in ["ebt", "baseline_transformer"]:
                return frames
            else:
                return frames, label
        except Exception as exception:
            log_path = self.ignore_filepath if self.ignore_filepath else self.potential_ignore_filepath
            handle_corrupt_file(exception, video_path, log_path)
