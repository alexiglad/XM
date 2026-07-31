from datetime import timedelta

import pytorch_lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import ToPILImage
import wandb
import gc

from data.vid.preencode_feature_extractor import preencode_dataset_features, hdf5_split_finished
from data.vid.kinetics_dataloader import *
from data.vid.something_dataloader import SomethingDataset
from data.vid.vid_synthetic_dataset import VIDSyntheticDataset

from data.img.imagenet_dataloader import *
from data.img.img_synthetic_dataloader import *
from utils.cache_latents import CachedLatentsDataset, cache_image_dataset

from data.nlp.pajama_dataloader import RedPajamaDataset
from data.nlp.fineweb_dataloader import FineWebDataset
from data.nlp.collator import NLP_HF_Collator
from data.nlp.bigbench_dataloader import BigBenchDataset
from data.nlp.gsm8k_dataloader import GSM8KDataset
from data.nlp.lambada_dataset import LambadaDataset
from data.nlp.squad_dataloader import SQuADDataset
from data.nlp.ai2arc_dataloader import AI2ArcDataset
from data.nlp.planbench_dataloader import PlanBenchDataset
from data.nlp.synthetic_dataset import NLPSyntheticDataset

from model.nlp.baseline_transformer import Baseline_Transformer_NLP
from model.nlp.mdlm import MDLM_NLP

from model.img.jumpy_cc import Jumpy_IMG_CC
from model.vid.jumpy_wm import Jumpy_VID_WM

from model.vid.dit_wm import DiT_VID_WM
from model.img.dit_cc import DiT_IMG_CC

from model.model_utils import center_crop_arr, temporary_ddp_timeout
from optimization import WarmUpCosineAnnealingLR
from utils.metrics_calculator import get_torchmetrics
import torchvision.transforms.functional as TF


class ModelTrainer(L.LightningModule):
    def __init__(self, hparams, trained_model = None):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        # self.txt_logger = hparams.txt_logger if txt_logger == None else txt_logger # txt_logger is no longer supported
        if self.hparams.modality == "VID": #is computer vision
            self.image_dims = self.hparams.image_dims # list size two
            if self.hparams.custom_image_normalization:
                self.transform = transforms.Compose([
                    transforms.Resize((self.image_dims[0], self.image_dims[1])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                normal_lookup = { #NOTE is std, mean
                    "k400": ([1.00370078, 0.99871626, 0.97407404], [-0.24295556, -0.24931058, -0.13959686]),
                    "smth": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
                    "ImageNet": ([1, 1, 1], [0, 0, 0])
                }

                normal_lookup["something"] = normal_lookup["smth"]
                normal_lookup["ImageNet1k"] = normal_lookup["ImageNet"]

                if self.hparams.dataset_name in normal_lookup:
                    std, mean = normal_lookup[self.hparams.dataset_name]
                    self.transform.transforms.append(transforms.Normalize(mean=mean, std=std))
                else:
                    raise ValueError(f"{self.hparams.dataset_name} not in normal lookup")
                    
            else:
                if self.hparams.backbone_type == "rae":
                    # RAE expects [0, 1] input and normalizes internally
                    self.transform = transforms.Compose([
                        transforms.Resize((self.image_dims[0], self.image_dims[1])),
                        transforms.ToTensor(),
                    ])
                elif self.hparams.vae_normalization:
                    if self.hparams.augment_video_data:
                        assert not self.hparams.preencode_dataset, "dont have support for augmentations with pre-encoded dataset yet"
                        self.transform = transforms.Compose([
                            transforms.Resize((self.image_dims[0], self.image_dims[1])),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ColorJitter(
                                brightness=0.1,
                                contrast=0.1,
                                saturation=0.1,
                                hue=0.02,
                            ),
                            transforms.Lambda(
                                lambda img: TF.adjust_gamma(img, gamma=random.uniform(0.9, 1.1))
                            ),  # slight random gamma shift
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                        ])
                    else:
                        self.transform = transforms.Compose([
                            transforms.Resize((self.image_dims[0], self.image_dims[1])),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                        ])
                else: # imagenet standardization
                    self.transform = transforms.Compose([
                        transforms.Resize((self.image_dims[0], self.image_dims[1])),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        if self.hparams.modality == "IMG": # using transform from DiT codebase https://github.com/facebookresearch/DiT
            img_transforms = [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.hparams.image_dims[0])),
                transforms.ToTensor(),
            ]
            if self.hparams.backbone_type == "vae":  # VAE expects [-1, 1]; RAE expects [0, 1] (normalizes internally)
                img_transforms.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True))
            elif self.hparams.backbone_type == "rae":
                pass
            else:
                raise NotImplementedError(f"have not yet implemented self.hparams.backbone_type: {self.hparams.backbone_type}")
            self.transform = transforms.Compose(img_transforms)

        self.to_pil = ToPILImage()
        self.full_ds = None
        
        if trained_model is not None:
            self.model = trained_model
        else:
            if self.hparams.model_name == "jumpy":
                if self.hparams.modality == "VID":
                    if self.hparams.video_task == "gc_world_model":
                        self.model = Jumpy_VID_WM(self.hparams)
                    else:
                        raise ValueError(f"task type: {self.hparams.video_task} not supported in base model trainer as a model as of now")
                elif self.hparams.modality == "IMG":
                    if self.hparams.image_task == "class_conditional":
                        self.model = Jumpy_IMG_CC(self.hparams)
                    else:
                        raise ValueError(f"image_task: {self.hparams.image_task} not supported for jumpy model as of now")
                else:
                    raise ValueError(f"Modality: {self.hparams.modality} not supported as a base model trainer model as of now")
            elif self.hparams.model_name == "baseline_transformer":
                if self.hparams.modality == "NLP":
                    self.model = Baseline_Transformer_NLP(self.hparams)
                else:
                    raise ValueError(f"Modality: {self.hparams.modality} not supported as a base model trainer model as of now")
            elif self.hparams.model_name == "dit":
                if self.hparams.modality == "VID":
                    if self.hparams.video_task == "gc_world_model":
                        self.model = DiT_VID_WM(self.hparams)
                    else:
                        raise ValueError(f"task type: {self.hparams.video_task} not supported for dit VID model as of now")
                elif self.hparams.modality == "IMG":
                    if self.hparams.image_task == "class_conditional":
                        self.model = DiT_IMG_CC(self.hparams)
                    else:
                        raise ValueError(f"image_task: {self.hparams.image_task} not supported for dit model as of now")
                elif self.hparams.modality == "NLP": # dit is a slight misnomer here -- this is MDLM (masked diffusion language model), ref implementation https://arxiv.org/pdf/2507.15857
                    self.model = MDLM_NLP(self.hparams)
                else:
                    raise ValueError(f"Modality: {self.hparams.modality} not supported as a base model trainer model as of now")
            else:
                raise ValueError(f"do not recognize model name: {self.hparams.model_name}")
        
        if self.hparams.compile_model:
            self.model = torch.compile(self.model, fullgraph=True)

        phases = ['train', 'valid', 'test']
        self.torchmetrics_dict = nn.ModuleDict()
        self.metrics = []
        for metric in self.hparams.metrics_list:
            self.metrics.append(metric)
        if len(self.metrics) > 0:
            assert self.hparams.num_classes != -1, "please set num_classes to the appropriate amount for the in use metrics. if are using accuracy and num_classes varies just set it to something that makes sense (shouldnt matter in that case)"
            assert self.hparams.metrics_task != "", "please set metrics_task to the appropriate value for your metrics"
        for phase in phases:
            for metric in self.metrics:
                self.torchmetrics_dict[f"{phase}_{metric}"] = get_torchmetrics(metric, self.hparams.metrics_average_type, self.hparams.num_classes, self.hparams.metrics_task)

        if self.hparams.wandb_watch:
            for name, module in self.model.named_modules(): # for activation logging
                module.name = name

        
    def on_train_start(self):
        if self.hparams.debug_unused_parameters: 
            for name, param in self.model.named_parameters():
                if param.requires_grad and "image_encoder" not in name: # NOTE you will need to modify this code to exclude specific frozen portions when debugging
                    print(f"registering param - {name}")
                    param.register_hook(self.create_hook(name))
                else:
                    self.model.parameters_not_to_check.add(name)

    def on_save_checkpoint(self, checkpoint): # stash wandb run id so resume_training_ckpt can resume the same wandb run
        if wandb.run is not None:
            checkpoint['wandb_run_id'] = wandb.run.id

    def create_hook(self, name): #this is only used for debugging with `debug_unused_parameters`
        def hook(grad):
            self.model.used_parameters.add(name)  # Adjusted to self.model.used_parameters
        return hook
    
    @staticmethod
    def wandb_activation_hook(run, step):
        """ Weights & Biases histogram activation hook. """
        def hook(module, input, output):
            if isinstance(output, tuple):
                pass # caused bug AttributeError: 'list' object has no attribute 'detach'
            else:
                run.experiment.log(
                    {f"activations/{module.name}": wandb.Histogram(output.detach().cpu().float())}, 
                    step=step
                )

        return hook
    
    def training_step(self, batch, batch_idx):
        if not self.hparams.no_wandb and self.hparams.wandb_watch and self.global_step % self.hparams.wandb_watch_log_freq == (self.hparams.wandb_watch_log_freq - 1): # activation logging
            hook_handles = []
            hook_function = self.wandb_activation_hook(run=self.logger, step=self.global_step)
            for module in self.model.modules():
                if any(param.requires_grad for param in module.parameters(recurse=False)): # only do for unfrozen params that are training
                    handle = module.register_forward_hook(hook_function)
                    hook_handles.append(handle)
            
            eval_step_dict = self.eval_step(batch, "train")
            for handle in hook_handles:
                handle.remove()

        else:
            eval_step_dict = self.eval_step(batch, "train")
        
        self.log_metrics(eval_step_dict, "train")
        return eval_step_dict['loss']   
    
    def on_after_backward(self):
        if self.hparams.log_gradients:
            total_norm = 0.0
            num_parameters = 0
            num_grads_exceeding_clip_val = 0
            total_gradients = 0 # this is different from num_parameters since .parameters is for tensors of params but doesnt count each invididual parameter
            for param in self.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm  # Add the norm value to the total sum
                    num_parameters += 1
                    
                    total_gradients += torch.numel(param.grad)
                    num_grads_exceeding_clip_val += torch.sum(param.grad.abs() > self.hparams.gradient_clip_val)
                    
            assert num_parameters > 0, "no gradients after backwards detected please investigate"
            average_norm = (total_norm / num_parameters).detach()
            percentage_clipped = ((num_grads_exceeding_clip_val / total_gradients) * 100).detach()              
            
            things_to_log = {} 
            things_to_log['avg_gradient_norms'] = average_norm
            things_to_log['pct_gradient_clipped'] = percentage_clipped
            self.log_metrics(things_to_log, "train", log_torchmetrics = False)

    def on_before_zero_grad(self, optimizer):
        if self.hparams.ema_model != 1.0:
            assert hasattr(self.model, 'ema_update'), "self.model needs an ema_update function in order to be able to ema model"
            self.model.ema_update()
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        #NOTE when using this may need to explicitly add code like 'if "image_encoder" not in name' for frozen params (with requires_grad == False)
        if self.hparams.debug_unused_parameters:
            all_parameters = {name for name, _ in self.model.named_parameters()}
            unused_parameters = all_parameters - self.model.used_parameters - self.model.parameters_not_to_check
            
            print(f"number of parameters total: {len(all_parameters)}")
            print(f"number of unused_parameters: {len(unused_parameters)}")
            print(f"Unused parameters: {unused_parameters}")
            print(f"Used parameters: {self.model.used_parameters}")
        
        if self.hparams.manual_gc_collect_every_n_steps != -1:
            if self.global_step % self.hparams.manual_gc_collect_every_n_steps == 0:
                print("calling GC manually")
                gc.collect()            
           
    def validation_step(self, batch, batch_idx):
        eval_step_dict = self.eval_step(batch, "valid")
        self.log_metrics(eval_step_dict, "valid")

    def test_step(self, batch, batch_idx):
        eval_step_dict = self.eval_step(batch, "test")
        self.log_metrics(eval_step_dict, "test")
            
    def eval_step(self, batch, phase):
        things_to_log = self.model.forward_loss_wrapper(batch, phase) # things_to_log will be a dict of various things being logged. it NEEDS TO contain the 'loss' key as this is used to backprop

        if len(self.metrics) > 0:
            raise NotImplementedError("Need to implement torchmetrics stuff, i.e. looping through self.torchmetrics_dict.keys(), checking to make sure 'phase in key', and updating based off predicted and labels i.e. self.torchmetrics_dict[key].update(logits, labels), more info https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html (just be careful make sure to detach logits before using them and only update current phase). recommended to possibly return things_to_log and logits from forward_loss_wrapper to do this easily")

        return things_to_log
    
    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self): # this is a PL hook that returns optimizer and lr scheduler
        if self.hparams.modality == "NLP":
            return self.configure_optimizers_nlp()
        elif self.hparams.modality == "VID":
            return self.configure_optimizers_vid()
        elif self.hparams.modality == "IMG":
            return self.configure_optimizers_img()
        else:
            raise NotImplementedError(f"Modality {self.hparams.modality} does not have configure optimizers supported yet")
        
    def get_optimizer(self, optimizer_parameters):
        assert self.hparams.optimizer == "adamw", f"unsupported optimizer: {self.hparams.optimizer}"
        return torch.optim.AdamW(optimizer_parameters, betas=[self.hparams.beta1, self.hparams.beta2])
    
    def on_warm_up_finished(self):
        if hasattr(self.model, 'warm_up_finished'):
            self.model.warm_up_finished()
            print("Warm up finished, calling self.model.warm_up_finished()")
        else:
            print("Warm up finished, no self.model.warm_up_finished() exists so not doing anything")
    
    def get_lr_scheduler(self, optimizer):
        cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_scheduling_steps - self.hparams.warm_up_steps, eta_min=self.hparams.peak_learning_rate / self.hparams.min_lr_scale)
        lr_scheduler = WarmUpCosineAnnealingLR(optimizer, warm_up_steps = self.hparams.warm_up_steps, warm_up_base_lr_divider = self.hparams.warm_up_base_lr_divider, cosine_scheduler=cosine_annealing_scheduler, warm_up_finished_func=self.on_warm_up_finished)
        return lr_scheduler
    
    def get_optimizer_scheduler_dict(self, optimizer_parameters):
        optimizer = self.get_optimizer(optimizer_parameters)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
            
    def configure_optimizers_nlp(self):
        if self.hparams.model_name in ("baseline_transformer", "dit"):
            all_params = [param for _, param in self.model.named_parameters()]
            optimizer_parameters = [
                {'params': all_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)

        else:
            raise NotImplementedError(f"havent implemented configure optimizers for model {self.hparams.model_name}")

        
    def configure_optimizers_vid(self):
        if self.hparams.model_name == "baseline_transformer":
            encoder_params = list(self.model.image_encoder.parameters())
            other_params = [param for name, param in self.model.named_parameters() if not name.startswith('ema') and not any(keyword in name for keyword in ['image_encoder'])]

            optimizer_parameters = [
                {'params': encoder_params, 'weight_decay': 0, 'lr': 0},
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)

        elif self.hparams.model_name in ("dit", "jumpy"):
            encoder_params = list(self.model.image_encoder.parameters())
            other_params = [param for name, param in self.model.named_parameters() if not name.startswith('ema') and not any(keyword in name for keyword in ['image_encoder'])]

            optimizer_parameters = [
                {'params': encoder_params, 'weight_decay': 0, 'lr': 0},
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)

        else:
            raise NotImplementedError(f"havent implemented configure optimizers for model {self.hparams.model_name}")

    def configure_optimizers_img(self):
        if self.hparams.model_name in ("dit", "jumpy"):
            other_params = [param for name, param in self.model.named_parameters() if not name.startswith('ema') and not any(keyword in name for keyword in ['image_encoder'])]

            optimizer_parameters = [
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
        else:
            raise NotImplementedError(f"havent implemented configure optimizers for model {self.hparams.model_name}")


    def create_full_ds(self):
        if self.hparams.dataset_name == "vid_synthetic":
            self.full_ds = VIDSyntheticDataset(self.hparams)
        elif self.hparams.dataset_name == "pajama":
            self.full_ds = RedPajamaDataset(self.hparams)
        elif self.hparams.dataset_name == 'fineweb':
            self.full_ds = FineWebDataset(self.hparams)
        elif "bigbench" in self.hparams.dataset_name:
            x = self.hparams.dataset_name
            self.full_ds = BigBenchDataset(self.hparams, "train", x[x.find('_') + 1 :])
        elif self.hparams.dataset_name == "planbench":
            self.full_ds = PlanBenchDataset(self.hparams, split = "train")
        elif self.hparams.dataset_name == "nlp_synthetic":
            self.full_ds = NLPSyntheticDataset(self.hparams)
        elif self.hparams.dataset_name == "img_synthetic":
            self.full_ds = ImgSyntheticDataset(self.hparams)
        else:
            raise NotImplementedError(f"haven't implemented dataset {self.hparams.dataset_name} full_ds yet")

    def setup(self, stage=None):
        # NOTE when passing stage into datasets/dataloaders use string rep not the stage param from this func since is a PL enum
        # Assign train/val datasets for use in dataloaders 
        assert self.hparams.test_split_pct == 0, "Haven't implemented nonzero value for test_split_pct yet"

        if stage == "fit":
            # all of these conditions need to have manual split
            if self.hparams.dataset_name in ["vid_synthetic", "pajama", "fineweb", "bigbench", "planbench", "nlp_synthetic", "img_synthetic"]:
                self.create_full_ds()
                train_samples = int(len(self.full_ds) * (1 - self.hparams.validation_split_pct))
                valid_samples = len(self.full_ds) - train_samples
                self.train_ds, self.val_ds = random_split(self.full_ds, [train_samples, valid_samples])
            elif self.hparams.dataset_name == 'k400':
                self.train_ds = Kinetics400Dataset(self.hparams, split = 'train', transform = self.transform)
                self.val_ds = Kinetics400Dataset(self.hparams, split = 'val', transform = self.transform)
            elif self.hparams.dataset_name in ('something' , 'smth'):
                self.train_ds = SomethingDataset(self.hparams, split = 'train', transform = self.transform)
                self.val_ds = SomethingDataset(self.hparams, split = 'val', transform = self.transform)
            elif self.hparams.dataset_name in ('imagenet' , 'imagenet1k'):
                if self.hparams.use_cached_img_latents:
                    with temporary_ddp_timeout(timedelta(hours=24)): # latent caching can take hours; non-rank-0 ranks would otherwise time out at the barrier
                        train_cache, val_cache = cache_image_dataset(self.hparams, self.transform, self.device)
                    self.train_ds = CachedLatentsDataset(str(train_cache))
                    self.val_ds = CachedLatentsDataset(str(val_cache))
                else:
                    self.train_ds = ImageNetDataset(self.hparams, split = 'train', transform = self.transform)
                    self.val_ds = ImageNetDataset(self.hparams, split = 'val', transform = self.transform)
            elif self.hparams.dataset_name == "gsm8k":
                self.train_ds = GSM8KDataset(self.hparams, split = "train")
                self.val_ds = GSM8KDataset(self.hparams, split = "test") # no val just test https://huggingface.co/datasets/openai/gsm8k
            elif self.hparams.dataset_name == "ai2arc":
                self.train_ds = AI2ArcDataset(self.hparams, split = 'train')
                self.val_ds = AI2ArcDataset(self.hparams, split = 'validation')
            elif self.hparams.dataset_name == "squad":
                self.train_ds = SQuADDataset(self.hparams, split = 'train')
                self.val_ds = SQuADDataset(self.hparams, split = 'validation')
            else:
                raise NotImplementedError("Haven't implemented this dataset yet")
            print(f"{self.hparams.dataset_name} length of train_dataset: {len(self.train_ds)} and val_dataset: {len(self.val_ds)}")

            if self.hparams.custom_limit_train_batches != 0.0: # only cut train set if memorizing
                if self.hparams.custom_limit_train_batches >= 1: # absolute number of samples
                    subset_size = int(self.hparams.custom_limit_train_batches)
                else: # fraction of dataset
                    subset_size = int(self.hparams.custom_limit_train_batches * len(self.train_ds))
                self.train_ds = Subset(self.train_ds, list(range(subset_size)))
                print(f"{self.hparams.dataset_name} NEW adjusted length of train_dataset: {len(self.train_ds)} and val_dataset: {len(self.val_ds)}")

        # Assign test dataset for use in dataloader(s)
        elif stage == "test":
            if self.hparams.dataset_name in ('kinetics400' , 'k400'):
                self.test_ds = Kinetics400Dataset(self.hparams, split = "test", transform = self.transform)
            elif self.hparams.dataset_name in ('something' , 'smth'):
                self.test_ds = SomethingDataset(self.hparams, split = "test", transform = self.transform)
            elif self.hparams.dataset_name in ('imagenet' , 'imagenet1k'):
                self.test_ds = ImageNetDataset(self.hparams, split = "test", transform = self.transform)
            elif self.hparams.dataset_name == "pajama": # for now am assuming test split == val split, so dont save train or full ds here, just to get val split
                full_ds = RedPajamaDataset(self.hparams)
                train_samples = int(len(full_ds) * (1 - self.hparams.validation_split_pct))
                test_samples = len(full_ds) - train_samples
                _, self.test_ds = random_split(full_ds, [train_samples, test_samples])
            elif self.hparams.dataset_name == "fineweb":
                raise NotImplementedError(f"haven't implemented fineweb dataset test split yet")
            elif "bigbench" in self.hparams.dataset_name:
                x = self.hparams.dataset_name
                self.test_ds = BigBenchDataset(self.hparams, "validation", x[x.find('_') + 1 :]) #use val for testing as Bigbench only has train/val
            elif self.hparams.dataset_name == "gsm8k":
                self.test_ds = GSM8KDataset(self.hparams, split="test")
            elif self.hparams.dataset_name == "lambada":
                self.test_ds = LambadaDataset(self.hparams, split="test")
            elif self.hparams.dataset_name == "squad":
                self.test_ds = SQuADDataset(self.hparams, split="validation") # no test split use val
            elif self.hparams.dataset_name == "planbench":
                raise NotImplementedError(f"no planbench test split")
            elif self.hparams.dataset_name == "ai2arc":
                self.test_ds = AI2ArcDataset(self.hparams, split = "test")
            else:
                raise NotImplementedError("haven't implemented this dataset yet")
            print(f"{self.hparams.dataset_name} length of test_ds: {len(self.test_ds)}")
        else:
            raise ValueError(f"Unknown stage: {stage}, please investigate")
        
        # -- if preencoding is enabled then run preencoding for train and validation datasets if they already don't exist
        if self.hparams.preencode_dataset:
            assert self.hparams.modality == "VID" and (self.hparams.dataset_name == "something" or self.hparams.dataset_name == "vid_synthetic"), "please either add support for other modalities and datasets or update this assert statement!"
			# check if dataset hdf5 file exists
            if self.hparams.dataset_name == "vid_synthetic":
                pass # do nothing for vid_synthetic
            else:
                train_finished = hdf5_split_finished(self.hparams.hdf5_file_path, "train")
                val_finished = hdf5_split_finished(self.hparams.hdf5_file_path, "val")
                if train_finished and val_finished:
                    print(f"preencoded dataset path {self.hparams.hdf5_file_path} found, loading this dataset...")
                else:
                    # Subset wrapping would shadow the use_hdf5 flip below (only sets attr on the wrapper, not the underlying SomethingDataset), so the dataloader would silently fall back to the raw-video path
                    assert not isinstance(self.train_ds, Subset), "--custom_limit_train_batches is incompatible with --preencode_dataset (Subset wrapper shadows the post-preencode use_hdf5 flip)"
                    # only rank 0 writes the hdf5; other ranks wait at the barrier then read it. skip splits that are already finished so we don't re-encode them (and so we don't feed already-encoded features back through the encoder for datasets whose __init__ took the hdf5 fast path)
                    with temporary_ddp_timeout(timedelta(hours=24)): # preencoding can take many hours; non-rank-0 ranks would otherwise time out at the barrier
                        if self.trainer.is_global_zero:
                            if not train_finished:
                                preencode_dataset_features(self.hparams, self.model.image_encoder, self.train_ds, self.hparams.hdf5_file_path, split="train")
                            if not val_finished:
                                preencode_dataset_features(self.hparams, self.model.image_encoder, self.val_ds, self.hparams.hdf5_file_path, split="val")
                            print("finished preencoding dataset")
                        self.trainer.strategy.barrier()
                    # all splits now finished; flip datasets onto the hdf5 read path for this run (avoids falling back to slow raw-video reads during first-run training)
                    self.train_ds.use_hdf5 = True
                    self.val_ds.use_hdf5 = True
    
    def get_collate_fn(self):
        collate_fn = None if not self.hparams.modality == "NLP" else NLP_HF_Collator(self.hparams) #NOTE this assumes all modalities except NLP DONT have collator, may not be true in the future
        if self.hparams.dataset_name == "nlp_synthetic": #NOTE this is a hack to get around the fact that synthetic dataset cant return real text and thus cant use collate_fn
            collate_fn = None
        return collate_fn
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size_per_device, num_workers=self.hparams.num_workers_per_device, persistent_workers=True, collate_fn = self.get_collate_fn(), pin_memory = True, drop_last = False, shuffle = not self.hparams.no_shuffle, prefetch_factor=self.hparams.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size_per_device, num_workers=self.hparams.num_workers_per_device, persistent_workers=True, collate_fn = self.get_collate_fn(), pin_memory = True, drop_last = False, shuffle = False, prefetch_factor=self.hparams.prefetch_factor)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size_per_device, num_workers=self.hparams.num_workers_per_device, persistent_workers=True, collate_fn = self.get_collate_fn(), pin_memory = True, drop_last = False, shuffle = False, prefetch_factor=self.hparams.prefetch_factor)

    def log_metrics(self, metrics_dict, phase, log_torchmetrics = True):
        # first log torchmetrics if there are any
        if log_torchmetrics and len(self.metrics) > 0:
            phase_dict = {key : value for key, value in self.torchmetrics_dict.items() if phase in key}
            self.log_dict(phase_dict, on_step = False, on_epoch = True) # for these always do on_epoch

        # log all other metrics in metrics_dict
        scalar_metrics = {}
        keys = list(metrics_dict.keys()) # Iterate over a copy of the keys to avoid modification issues during iteration
        for key in keys:
            value = metrics_dict[key]
            if 'image' in key: # images
                image = self.to_pil(value)
                wandb_image = wandb.Image(image, mode="RGB")
                self.logger.experiment.log({f'{phase}_{key}': wandb_image}, step=self.global_step)

            elif 'video' in key: # videos
                video_np = value.cpu().numpy()
                assert video_np.ndim != 5, "video should not include batch dimension, either fix that or add support"
                if video_np.shape[1] in [1, 3]:
                    pass  # Axes are already correct
                elif video_np.shape[-1] in [1, 3]:
                    # If video_np is (frames, height, width, channels), transpose axes
                    video_np = video_np.transpose(0, 3, 1, 2)
                else:
                    raise ValueError(f"Unexpected video shape: {video_np.shape}")
                if video_np.dtype != np.uint8:
                    video_np = (video_np * 255).astype(np.uint8)
                assert self.hparams.time_between_frames != 1.0, "need to add support for logging with fps when using a different frame selection mechanism"
                fps = 1 / self.hparams.time_between_frames
                wandb_video = wandb.Video(video_np, fps=fps, format="mp4")
                self.logger.experiment.log({f'{phase}_{key}': wandb_video}, step=self.global_step)

            elif isinstance(value, torch.Tensor) and value.numel() > 1: # histogram
                self.logger.experiment.log({f"{phase}_{key}": wandb.Histogram(value.detach().cpu())}, step=self.global_step)

            elif isinstance(value, torch.Tensor) and value.dim() == 0: # two types of scalar, tensor (here) and int/float (below)
                scalar_metrics[f"{phase}_{key}"] = value.detach()
            elif isinstance(value, (int, float)):
                scalar_metrics[f"{phase}_{key}"] = value
            else:
                raise ValueError(f"unsupported type/format in log_metrics, type:, {type(value)}, key: {key}")

        if scalar_metrics:
            self.log_dict(scalar_metrics, sync_dist=True, prog_bar=True)
        
        if len(self.trainer.optimizers) == 0:
            pass # is during testing lr doesnt matter
        else:
            current_lr = self.trainer.optimizers[0].param_groups[-1]['lr'] # relies on lr for most of model being the last param
            self.log("Global_LR", current_lr, sync_dist=True)