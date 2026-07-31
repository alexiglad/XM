import os
from argparse import ArgumentParser
import time
import pytorch_lightning as L
from pytorch_lightning.strategies import DDPStrategy
import random
from datetime import datetime
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import sys
import wandb
import ast
import math
from tqdm import tqdm
import json
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from utils.dataloader_debugger import debug_dataloader
from model.model_utils import (
    load_trained_pl_model,
    model_sizes,
    init_wandb_watch,
    setup_wandb,
    format_lr,
    format_run_name_hparams,
    load_wandb_run_id,
    DEFAULT_DDP_TIMEOUT,
)
from utils import text_logger
from base_model_trainer import *

from inference.img.online_fid_eval_callback import ImageFIDCallback
from inference.nlp.online_lm_eval_callback import LM_Eval_Callback
from inference.vid.online_fvd_eval_callback import VideoFVDCallback

# note that much of this code is from the EBT codebase: https://github.com/alexiglad/ebt

def main(args):
    # set run name based on prefix as well as run_name_hparams
    base = f"{args.model_name}-{args.model_size}"
    if args.run_prefix:
        base += f"-{args.run_prefix}"
    parts = [base]
    if args.run_name_hparams:
        parts.extend(format_run_name_hparams(args.run_name_hparams, args))
    parts.append(f"bs={args.effective_batch_size}")
    parts.append(f"lr={format_lr(args.peak_learning_rate)}")
    args.run_name = ",".join(parts) # comma is safe for shells/paths/filenames/regex, and for passing ckpt paths back via --resume_training_ckpt
    args.wandb_run_name = args.run_name.replace(",", ";") # ; is more visually distinct in the wandb UI; only used for wandb display

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        os.system(f'scontrol update JobId={slurm_job_id} Name="{args.run_name}"')

    if(args.is_random_seed):
        seed_everything(random.randint(0,1000000), workers=True)
    else:
        seed_everything(33, workers=True) #33 is default
        
    if args.debug_mode:
        args.no_wandb = True
        args.detect_anomaly = True
        args.limit_train_batches = 1

    os.makedirs("./logs", exist_ok=True)

    wandb_logger = None
    if not args.no_wandb: # put this early so can capture text logs later
        resume_run_id = None
        if args.resume_training_ckpt:
            resume_run_id = load_wandb_run_id(args.resume_training_ckpt)
            if resume_run_id is None:
                print(f"WARNING: no wandb_run_id in ckpt {args.resume_training_ckpt}; will start a new wandb run")

        run = None # need both lines since setup_wandb is a @rank_zero_only function
        run = setup_wandb(args, resume_run_id=resume_run_id)

        wandb_logger = WandbLogger(save_dir="logs/", name=f'{args.wandb_run_name}', entity=f'{args.wandb_entity}', project=f'{args.wandb_project}', offline = args.wandb_offline, experiment=run)
        if args.wandb_tags != None:
            wandb_logger.experiment.tags = args.wandb_tags
    else:
        console_log_file_path = os.path.join("./logs", args.console_log_filename)
        sys.stdout = text_logger.Tee(sys.stdout, console_log_file_path)
        sys.stderr = text_logger.Tee(sys.stderr, console_log_file_path)
        print("$$$$$$$$$ NOTE THAT NOT ALL STDOUT LOGS (i.e. pytorch lighting logs) ARE CAPTURED THROUGH CONSOLE LOGGER, THIS IS ONLY RECOMMENDED FOR DEBUGGING $$$$$$$$$$")

    if args.is_slurm_run:
        if not args.override_slurm_checks:
            assert args.debug_mode == False and args.detect_anomaly == False and args.limit_train_batches == 1 and args.limit_val_batches == 1 and args.limit_test_batches == 1 and args.find_unused_parameters == False and args.debug_unused_parameters == False, "for slurm run cannot have certain params set to values since am assuming are not debugging, please check values here"

        print("Current Slurm job ID:", os.environ.get('SLURM_JOBID'))
        print("Current Slurm node list:", os.environ.get('SLURM_NODELIST'))
        print("SLURM_NTASKS:", os.environ.get("SLURM_NTASKS"))
        print("SLURM_GPUS_PER_NODE:", os.environ.get("SLURM_GPUS_PER_NODE"))
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # txt_logger is no longer supported [deprecated]
    # txt_logger = text_logger.setup_custom_logger(log_filename = args.debug_log_filename, print_console = True) # this used to be set to args.print_logs but is default true now and print_logs is not used
    # args.txt_logger = txt_logger

    if args.model_size != "": # set params based off of model size
        args.num_transformer_blocks = model_sizes[args.model_size]['num_transformer_blocks']
        args.multiheaded_attention_heads = model_sizes[args.model_size]['multiheaded_attention_heads']
        args.embedding_dim = model_sizes[args.model_size]['embedding_dim']
        print("model_size", args.model_size, "args.num_transformer_blocks", args.num_transformer_blocks, "args.multiheaded_attention_heads", args.multiheaded_attention_heads, "args.embedding_dim", args.embedding_dim)

    if args.override_embedding_dim > 0:
        args.embedding_dim = args.override_embedding_dim

    if args.override_transformer_blocks > 0:
        args.num_transformer_blocks = args.override_transformer_blocks

    # hparam assertion sanity checks/modality specific logic
    if args.modality == "VID":
        if args.backbone_type == "dinov2":
            assert args.embedding_dim == 0, "embedding dim defined implicitly by encoder dimensionality"
            vit_backbone_embed_dim_map = {"small": 384, "base": 768, "large": 1024, "giant": 1536} # gotten from dinov2 repo
            args.vit_backbone_dim = vit_backbone_embed_dim_map[args.vit_backbone_size]
            args.embedding_dim = args.vit_backbone_dim
        elif args.backbone_type == "vae" or args.backbone_type == "rae":
            assert args.embedding_dim != 0, "must define embedding dim for vae"
        else:
            raise NotImplementedError(f"Unspported backbone type: {args.backbone_type}")
    elif args.modality == "NLP":
        assert args.embedding_dim != 0, "must define embedding dim for NLP models"
    elif args.modality == "IMG":
        args.cached_img_latents_dir = f"{os.getenv('HF_HOME')}/cached_latents/{args.dataset_name}" if args.cached_img_latents_dir is None else args.cached_img_latents_dir
        args.backbone_type = "vae" if args.backbone_type != "rae" else args.backbone_type # always uses vae unless rae is specified
        assert args.embedding_dim != 0, "must define embedding dim for IMG models"
        if args.backbone_type == "rae":
            assert args.image_dims[0] >= 256 and args.image_dims[1] >= 256, "RAE requires image_dims >= 256x256 since it decodes to 256x256"
        if args.use_ot_flow:
            assert args.xm_best_of_k == 1, "--use_ot_flow is the OT-CFM baseline; it must be compared against XMs, so xm_best_of_k must be 1"
            assert args.diffusion_supervision_type == "velocity", "--use_ot_flow is only meaningful for flow matching (velocity supervision)"
            assert args.model_name == "dit" and args.image_task == "class_conditional", "--use_ot_flow is only wired up for dit + class_conditional (IMG)"
    else:
        raise ValueError(f"please add support for modality {args.modality}")
    
    args.num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES', 1)) # may not exist if not using slurm so default to 1; multi node only supports slurm as of now
    print(f"SLURM_JOB_NUM_NODES: {args.num_nodes}")
    print("torch.cuda.device_count()", torch.cuda.device_count())
    assert args.num_nodes == 1 or args.gpus == "-1", "multi-node requires --gpus '-1' (use all GPUs); specifying a subset is not supported"
    if args.gpus == "-1":
        num_gpus = args.num_nodes * torch.cuda.device_count()
    elif '[' in args.gpus: # is a list
        num_gpus = len(args.gpus.split(","))
        args.gpus = json.loads(args.gpus)
    else:
        num_gpus = int(args.gpus)
    print("devices/args.gpus: ", args.gpus)

    args.total_num_workers = args.num_workers_per_device * num_gpus
    print("num_nodes", args.num_nodes, "total num_workers across all GPUs", args.total_num_workers, "num workers per GPU", args.num_workers_per_device, "num_GPUs", num_gpus)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    assert (device == torch.device('cuda') and num_gpus > 0), "using cpu instead of cuda. if you would like to proceed please remove this line and change code below to not use GPUs, otherwise check packages to ensure torch/others have cuda support"
    print(f'GPU Availability: {device}, gpus: {num_gpus}\n')
    args.num_gpus = num_gpus
    assert args.effective_batch_size > 0, "must set --effective_batch_size (> 0)"
    assert args.effective_batch_size % (args.batch_size_per_device * num_gpus) == 0, \
        f"effective_batch_size ({args.effective_batch_size}) must be divisible by bs_per_device * num_gpus ({args.batch_size_per_device * num_gpus})"
    assert args.accumulate_grad_batches == -1, "accumulate_grad_batches is computed automatically and cannot be set manually"
    args.accumulate_grad_batches = args.effective_batch_size // (args.batch_size_per_device * num_gpus)
    print(f"effective_batch_size: {args.effective_batch_size}", "batch_size_per_device", args.batch_size_per_device, "accumulate_grad_batches", args.accumulate_grad_batches)
    if args.lr_scaling_rule:
        scaled_lr = args.peak_learning_rate * args.effective_batch_size / 256 
        args.peak_learning_rate = scaled_lr
        print(f"Learning Rate rescaled to: {scaled_lr} based off lr_scaling_rule")
    if args.max_scheduling_steps == -1:
        args.max_scheduling_steps = args.max_steps

    model_trainer = ModelTrainer(args)

    if args.execution_mode == "finetune":
        assert args.finetuning_model_ckpt != None and args.resume_training_ckpt == "", "Must provide a checkpoint when finetuning and cannot provide a resume_training_ckpt."
        model_trainer.model = load_trained_pl_model(args.finetuning_model_ckpt, args)

    timestamp = int(time.time())
    dt_object = datetime.fromtimestamp(timestamp)
    dt_string = dt_object.strftime("%Y-%m-%d_%H-%M-%S")

    if args.debug_dataloader:
        debug_dataloader(args, model_trainer)
        return
    if args.log_model_archi:
        print(str(model_trainer.model))
        print(str(args))

    if not args.no_wandb and args.wandb_watch:
        init_wandb_watch(wandb_logger, model_trainer, args.wandb_watch_log_freq)

    print(f'pytorch version: {torch.__version__}\n')

    if args.set_matmul_precision is not None: #default is highest
        torch.set_float32_matmul_precision(args.set_matmul_precision)
    
    callbacks = []

    checkpoint_filename = "epoch={epoch}-step={step}-" + args.checkpoint_monitor_string + "={"+args.checkpoint_monitor_string+":.4f}"
    extra_kwargs = {}
    checkpoint_callback = ModelCheckpoint(monitor=args.checkpoint_monitor_string, mode = args.checkpoint_monitor_mode, save_top_k=args.save_top_k_ckpts, save_last = True, dirpath=f"{args.checkpoint_base_dir}/{args.run_name}_{dt_string}", filename=checkpoint_filename, verbose=True, **extra_kwargs)
    
    callbacks.append(checkpoint_callback)

    if args.run_online_evaluation:
        if args.run_online_fid_eval_every_n_epochs > 0:
            if args.modality == "IMG" and args.image_task == "class_conditional":
                online_eval_callback = ImageFIDCallback(args)
                print("Online FID Image Callback is set up")
                callbacks.append(online_eval_callback)
            else:
                raise NotImplementedError("have not yet implemented callbacks for specified modality and task, please implement")
        elif args.run_online_fvd_eval_every_n_epochs > 0:
            if args.modality == "VID":
                online_eval_callback = VideoFVDCallback(args)
                print("Online FVD Video Callback is set up")
                callbacks.append(online_eval_callback)
            else:
                raise NotImplementedError("run_online_fvd_eval_every_n_epochs is only supported for VID modality")
        elif args.run_online_lm_eval_every_n_val > 0:
            online_eval_callback = LM_Eval_Callback(args)
            print("Online LM Eval Callback is set up")
            callbacks.append(online_eval_callback)
        else:
            raise NotImplementedError("have not implemented online evaluation for type desired")
    
    for name, param in model_trainer.model.named_parameters():
        if not param.requires_grad:
            print(f"Non-trainable parameters: {name} with shape {param.shape}")
    
    if not args.only_test: #training and testing (if testing selected) as per usual
        print("$$$$$$$$$$  STARTED TRAINING  $$$$$$$$$$")
        trainer = set_trainer(args, wandb_logger, callbacks)
        resume_training_ckpt = None if args.resume_training_ckpt == "" else args.resume_training_ckpt
        trainer.fit(model_trainer, ckpt_path=resume_training_ckpt)
        
        if args.run_testing_after_training:
            args.only_test_model_ckpt = checkpoint_callback.best_model_path
            best_model = ModelTrainer.load_from_checkpoint(
                checkpoint_callback.best_model_path, 
                hparams=args
            )
            clear_cache()
            print(f"best model path that will be used during testing {checkpoint_callback.best_model_path}")
            print("$$$$$$$$$$  STARTED TESTING AFTER TRAINING  $$$$$$$$$$")
            raise NotImplementedError("need to test this with newer PL")
            # test_trainer = L.Trainer(logger=wandb_logger,devices=1,num_nodes=1) NOTE tested this code does not work gets stuck, TODO test with newer PL
            #warning reference - https://github.com/Lightning-AI/lightning/issues/12862
            best_model.eval()
            trainer.test(best_model)
        clear_cache()
    else: #only testing
        print("$$$$$$$$$$  ONLY TESTING MODEL ([NO]) TRAINING  $$$$$$$$$$")
        assert args.only_test_model_ckpt != None, "Must supply pretrained model when only testing"
        checkpoint = torch.load(args.only_test_model_ckpt, weights_only=False)
        pretrained_hparams = checkpoint['hyper_parameters']

        default_args = vars(args).copy() # NOTE this is so can test older models trained with older code as well as use newer hparams (for inference) on pretrained models, may be finicky feel free to tweak
        for key, value in default_args.items():
            if key not in pretrained_hparams:
                pretrained_hparams[key] = value
                print(f"MISSING PARAMETER IN PRETRAINED CHECKPOINT: Using args set value for missing parameter in pretrained checkpoint '{key}': {value}")
            if key.startswith("infer_"):
                pretrained_hparams[key] = value
                print(f"OVERRIDING PARAMETER IN PRETRAINED CHECKPOINT FOR INFERENCE: Using args set value for inference parameter '{key}': {value}")

        model = ModelTrainer(pretrained_hparams)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model_trainer = ModelTrainer(args, trained_model=model.model) # need to use args as we have to use the model most recently passed in for inference

        trainer = set_trainer(args, wandb_logger, callbacks, stage = "test")
        model_trainer.model.eval()
        trainer.test(model_trainer)

def set_trainer(args, wandb_logger, callbacks, stage = "train"):
    torch.autograd.set_detect_anomaly(args.detect_anomaly) #NOTE seems pl detect anomaly is not working so manually set it here

    ddp_timeout = DEFAULT_DDP_TIMEOUT # slow sections (preencoding, online eval) bump this via temporary_ddp_timeout instead
    if args.find_unused_parameters:
            args.distributed_strategy = DDPStrategy(find_unused_parameters=True, timeout=ddp_timeout)
    elif args.distributed_strategy == 'ddp':
            args.distributed_strategy = DDPStrategy(timeout=ddp_timeout)
    args.overfit_batches = int(args.overfit_batches) if int(args.overfit_batches) == args.overfit_batches else args.overfit_batches
    profiler = None if args.profiler == "" else args.profiler
    gradient_clip_val = args.gradient_clip_val if args.gradient_clip_val > 0 else None
    limit_val_batches = 0 if args.overfit_batches > 0 else args.limit_val_batches
    val_check_interval = args.val_check_interval if args.val_check_interval == 1.0 else args.val_check_interval * args.accumulate_grad_batches  #NOTE the reason we mult by args.accumulate_grad_batches is because of this bug https://github.com/Lightning-AI/pytorch-lightning/issues/12205
    limit_test_batches = args.limit_test_batches if args.limit_test_batches == 1 else args.limit_test_batches * args.accumulate_grad_batches
    trainer = L.Trainer(
        accelerator="auto",
        devices = args.gpus,
        num_nodes=args.num_nodes,
        precision=args.float_precision,
        max_steps=args.max_steps,
        logger=wandb_logger,
        enable_model_summary=args.log_model_archi,
        callbacks = [*callbacks, ModelSummary(max_depth=-1)],
        strategy = args.distributed_strategy, 
        enable_checkpointing=True,
        fast_dev_run = args.fast_dev_run,
        num_sanity_val_steps = args.val_sanity,
        limit_train_batches = args.limit_train_batches,
        limit_val_batches = limit_val_batches,
        limit_test_batches = limit_test_batches,
        detect_anomaly=args.detect_anomaly,
        gradient_clip_val=gradient_clip_val,
        overfit_batches=args.overfit_batches,
        profiler=profiler,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.deterministic,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        inference_mode=False # set inference mode to false so models can take input grads during testing if needed
    )
    return trainer
    

def clear_cache():
    torch.cuda.empty_cache()
    

def get_parser():
    parser = ArgumentParser()

    #SUPER IMPORTANT HPARAMS--always set these properly #############################################
    
    parser.add_argument("--run_prefix", help="prefix appended after model_name-model_size in run name", default="")

    parser.add_argument("--run_name_hparams", help="hparams for run name (e.g. 'img=image_dims,ps=patch_size'); each value after an equal sign is looked up for arg values when possible, otherwise the literal value is used, delineate args with a comma; always appends effective batch size and peak learning rate", nargs="*", default=[])

    parser.add_argument("--modality", help="is the model being trained for NLP, VID, IMG. default is VID, and most of the code has been designed around VID.", type=str, default="VID")

    parser.add_argument("--model_name", help="model_type/name", default='dit')

    parser.add_argument("--model_size", help="model size, is used to set num_transformer_blocks, multiheaded_attention_heads, and embedding_dim", choices=model_sizes.keys(), default='xxs')

    #NOTE not all below hparams are implemented for all models, so please check before using one     

    # VID SPECIFIC PARAMS ############################################################################################################

    parser.add_argument("--video_task", help="task for video modality, for now supports goal conditioned world model (gc_world_model)", choices=["gc_world_model"], type=str, default="gc_world_model")

    parser.add_argument("--vit_backbone_size", help="small, base, large or giant", type=str, default="base")

    parser.add_argument("--backbone_type", help="backbone type for VID encoder", choices=["dinov2", "vae", "rae"], type=str, default="vae")

    parser.add_argument("--time_between_frames", help="time (seconds) between frames of video", type=float, default=0.25)
    
    parser.add_argument("--sampling_rate", help="how many frames to draw a single video frame from. e.g. sampling rate=4 will have three frames in between each sample. overrides time_between_frames", type=float, default=0.0)
    
    parser.add_argument("--use_raw_framerate", help="whether to use dataset default framerate with a sampling rate of 1. this overrides the time_between_frames and sampling_rate parameters.", action="store_true", default=False)

    parser.add_argument("--vae_normalization", help="whether to use the commonly used vae normalization of [-1, 1].", action="store_true", default=False)

    parser.add_argument("--sdxl_vae_standardization", help="whether to standardize vae encodings according to sdxl by multiplying by 0.18215.", action="store_true", default=False)

    parser.add_argument("--weight_tie_vae_proj", help="for baseline transformer, weight ties the projection to vae latent space", action="store_true", default = False)

    parser.add_argument("--world_modeling_given_frames", help="given frames for world modeling, defaults to first 3 and final so simulates goal-conditioned world models which have become popular; needs to be within context length; 0 indexed", nargs='+', type=int, default = [0, 1, 2, 15])

    parser.add_argument("--gc_world_model_unconditional", help="for VID gc_world_model: fully unconditional video generation. treats world_modeling_given_frames as empty so every frame is denoised/predicted at training and inference; cfg_dropout_prob and cfg_scale become no-ops", action="store_true", default=False)

    parser.add_argument("--log_video_every_n_steps", help="logs a generated video every n training steps", type=int, default=10000)

    parser.add_argument("--preencode_dataset", help="whether to encode the dataset and save it for faster loading later. NOTE: this will take a while and requires a lot of disk space. also, if you change any hparams related to the dataloading you might need to reencode the dataset", action="store_true", default=False)

    parser.add_argument("--hdf5_file_path", help="path to preencoded dataset hdf5 file", type=str, default="")

    parser.add_argument("--video_transformer_3d", help="whether to use an actual video transformer with 3d attention over space/time, rather than a more simplified flatten an image and use that as an embedding transformer (which equates to patch_size of vit = actual vae encoded image patch size).", action="store_true", default=False)

    parser.add_argument("--augment_video_data", help="whether or not to augment your video dataset with horizontal flip and slight ", action="store_true", default=False)

     
    # NLP SPECIFIC PARAMS ############################################################################################################

    parser.add_argument("--tokenizer", help="tokenizer for nlp tasks", type=str, default="EleutherAI/gpt-neox-20b")
    
    parser.add_argument("--pretokenize_dataset", help="whether to pretokenize the dataset and save that or just tokenize as loading the dataset. tokenizing the dataset takes a long time and may need to be done using debug dataloader. due to a bug with HF does not always work reliably, may get stuck. not currently implemented for fine-tuning datasets", action="store_true", default=False)

    parser.add_argument("--run_online_fvd_eval_every_n_epochs", help="runs FVD evaluation on val set at the specified every n epochs rate (VID modality only)", type=int, default=0)

    parser.add_argument("--online_fvd_eval_num_videos", help="number of videos when running FVD online evaluation", type=int, default=10000)

    parser.add_argument("--run_online_lm_eval_every_n_val", help="run online language model evaluation every n validations (dont wait whole epoch to val for LMs usually)", type=int, default=0)

    parser.add_argument("--online_lm_eval_tasks", help="tasks seperated by commas to run lm eval with", type=str, default="arc_easy,hellaswag,piqa,openbookqa,sciq")

    # IMG SPECIFIC PARAMS ########################################################################################################################

    parser.add_argument("--image_task", help="task for image modality", choices=["class_conditional"], type=str, default="class_conditional")

    parser.add_argument("--patch_size", help="patch size from DiT paper, larger means less compute", choices=[1, 2, 4, 8, 16, 32], type=int, default=2)

    parser.add_argument("--final_layer_init_std", help="std for re-initializing the DiT final layer with high variance for more spread-out outputs; 0 means use default DiT zero-init", type=float, default=0.0)

    parser.add_argument("--log_image_every_n_steps", help="logs a generated image every n training steps", type=int, default=500)

    parser.add_argument("--cfg_dropout_prob", help="dropout probability for conditioning (class labels or video conditioning frames) for classifier-free guidance", type=float, default=0.1)

    parser.add_argument("--run_online_fid_eval_every_n_epochs", help="runs only FID evaluation at the specified every n epochs rate", type=int, default=0)

    parser.add_argument("--online_fid_eval_num_samples", help="number of samples when running FID online evaluation", type=int, default=50000)

    parser.add_argument("--online_fid_cfg_scales", help="CFG scale for online FID evaluation", type=float, nargs='+', default=[0.0])

    # IMG LATENT CACHING ##################################################################

    parser.add_argument("--use_cached_img_latents", help="use pre-cached VAE latents instead of encoding images on-the-fly", action="store_true", default=False)
    
    parser.add_argument("--cached_img_latents_dir", help="directory containing cached latent files (defaults to LATENTS_CACHE_DIR from .env)", type=str, default=None)

    # LATENT CONDITIONING (for AdaLN) ##################################################################
    parser.add_argument("--use_latent_conditioning", help="enable latent conditioning added to class embedding for AdaLN", action="store_true", default=False)
    parser.add_argument("--latent_dim", help="dimension of latent z for conditioning", type=int, default=64)
    parser.add_argument("--latent_std", help="std of latent z ~ N(0, std^2)", type=float, default=1.0)
    parser.add_argument("--latent_scale", help="scale applied to projected latent before adding to class embedding", type=float, default=1.0)

    # DIFFUSION ################################################################
    parser.add_argument("--diffusion_steps", help="number of steps in diffusion", type = int, default=1000)

    parser.add_argument("--use_deterministic_reverse", help="makes it so the reverse process is deterministic for generation and uses ddim instead of ddpm with learned sigma", action="store_true", default = False)

    parser.add_argument("--learn_sigma", help="whether or not sigma should be learned for diffusion, check create_diffusion for more info", action="store_true", default = False)

    parser.add_argument("--diffusion_supervision_type", help="supervision type for diffusion (score or velocity / flow matching). score not suggested and is no longer supported everywhere (also gets worse performance, see SiT paper)", choices=["score", "velocity"], type=str, default="velocity")

    parser.add_argument("--fvd_inference_diffusion_steps", help="number of DDPM/flow steps used when generating videos for FVD evaluation", type=int, default=100)

    parser.add_argument("--online_fvd_cfg_scales", help="CFG scale(s) for online FVD evaluation; 0.0 = no CFG", type=float, nargs='+', default=[0.0])

    # FLOW specific ode
    parser.add_argument('--ode_method', help='', default='heun2', type=str)

    parser.add_argument('--ode_step_size', help='', default=0.01, type=float)

    parser.add_argument("--use_ot_flow", help="OT-CFM baseline: pair noise/data within each minibatch via exact EMD coupling (torchcfm default; sampled from plan with replacement). currently only wired up for IMG class-conditional dit with velocity supervision", action="store_true", default=False)

    # EXPLORATIVE MODELING ################################################################

    parser.add_argument("--xm_best_of_k", help="Explorative Modeling best-of-k for computing loss", type = int, default=1)

    parser.add_argument("--xm_chunk_bs_mult", help="Explorative Modeling chunk bs multiplier when doing best-of-k exploration for min loss, needs to be >= 1", type = int, default=4)

    parser.add_argument("--xm_save_mem_mode", help="whether or not the best-of-k exploration should be computed initially without necessary grads to reduce mem so one can increase the xm_chunk_bs_mult", action="store_true", default = False)

    parser.add_argument("--xm_debug_mode", help="whether or not to debug the Explorative Modeling loss using an assert statement", action="store_true", default = False)

    parser.add_argument("--xm_lm_embed_pct", help="fraction (0-1) of each token's embedding that is a per-position Explorative Modeling best-of-k embedding (baseline_transformer NLP only). when >0, replaces the prepend-token Explorative Modeling mode with a per-position concat scheme similar to MDLM; requires xm_best_of_k > 1", type = float, default=0.0)

    parser.add_argument("--infer_mdlm_rand_xm_mask", help="at MDLM inference, pick a random (but input-deterministic) mask token index instead of always using 0", action="store_true", default = False)
    
    parser.add_argument("--infer_mdlm_ar", help="at MDLM inference, predict one token at a time (left to right) instead of all masked tokens at once", action="store_true", default = False)

    #JUMPY GENERATIVE MODELS ##################################################################
    # generalize direct e2e prediction as well as flow/diffusion by allowing interpretation between them

    parser.add_argument("--jumpy_n_jumps", help="number of jumps for generation. 1 = standard end-to-end prediction. >1 chops generation into K steps along a linear interpolation from noise to data, trained by uniformly sampling a random jump index per element each batch", type=int, default=1)

    parser.add_argument("--jumpy_pred_type", help="prediction type for jumping: direct predicts the clean data x_1 (x-prediction, arXiv:2511.13720), velocity predicts (data - noise) and scales by 1/K per step (equivalent to flow matching velocity). NOTE direct with >20 inference steps leaves ~1-2% residual noise, as the t_eps clamp caps the final step below 1 (same behavior as regular JiT); velocity is exact at any step count", type=str, choices=["velocity", "direct"], default="velocity")

    parser.add_argument("--jumpy_loss_reweighting", help="weight the direct (x-pred) loss by 1/(1-t)^2, equivalent (for mse) to JiT's v-space loss (arXiv:2511.13720); requires jumpy_pred_type=direct", action="store_true", default=False)

    parser.add_argument("--jumpy_inference_steps", help="number of inference steps for jumping; 0 means same as jumpy_n_jumps. can differ from jumpy_n_jumps to scale jump size at inference", type=int, default=None)
    
    parser.add_argument("--jumpy_time_conditioning", help="pass discrete jump index k as explicit input to the model at each jump step", action="store_true", default=False)

    parser.add_argument("--jumpy_time_embed_cont", help="use continuous time embedding (k/K float in [0,1)) like flow matching instead of discrete integer indices", action="store_true", default=False)

    #MODEL AND ARCHITECTURE ##################################################################

    # transformer specific ############################################

    parser.add_argument("--context_length", help="context length for AR models, i.e. for language model (commonly 256) or video model (commonly 16)", type=int, default=0)
          
    parser.add_argument("--num_transformer_blocks", help="number of transformer blocks, uses default from model size specified", type=int, default=12)
    
    parser.add_argument("--multiheaded_attention_heads", help="number of attention heads for transformer, uses default from model size specified", type=int, default=2)

    parser.add_argument("--cross_attention_transformer", help="whether transformer has cross attention, note that not all transformers have support for this", action="store_true", default = False)

    ######################################################################

    parser.add_argument("--embedding_dim", help="embedding dimension for transformers, if are using a model size this is automatically set", type=int, default=384)

    parser.add_argument("--ffn_dim_multiplier", help="how much wider than the embedding dim the transformer FFN dim should be", type=float, default=4.0)

    parser.add_argument("--override_embedding_dim", help="override the embedding dimension for a transformer. do not use unless you know what you are doing since it will likely cause issues and mess up scaling trends", type=int, default=0)

    parser.add_argument("--override_transformer_blocks", help="override the number of blocks for a transformer. do not use unless you know what you are doing since it will likely cause issues and mess up scaling trends", type=int, default=0)

    parser.add_argument("--num_modality_processing_mlp_layers", help="number of linear layers in the MLP for processing each modality, need at least 1", type=int, default=1) # NOTE not currently used or implemented

    parser.add_argument("--weight_initialization_method", help="xavier or he", type=str, default="xavier")

    parser.add_argument("--weight_initialization_gain", help="gain of the weight init, see https://pytorch.org/docs/stable/nn.init.html, default is equiavalent to linear gain which is 1", type=float, default=1.0)

    parser.add_argument("--ema_model", help="ema of a copy of the model--this is commonly used for generative modeling for better performance or for SSL. 1.0 means dont use, should be <= 1, this is the decay", type=float, default=1.0)

    # seperate MLP specific #####################################################################
    # NOTE, these are not for transformer MLP are for seperate MLPs
    parser.add_argument("--mlp_dropout", help="dropout in basic MLP", type=float, default=0.0)
                
    parser.add_argument("--mlp_layer_norm", help="layer normalization", action="store_true", default=False)

    parser.add_argument("--mlp_dim_multiplier", help="how much wider than the embedding dims the final layer projector should be", type=float, default=2.0)

    parser.add_argument("--mlp_hidden_layers", help="number of linear layers in MLP after encoder, if is 1 is just a linear layer", type=int, default=0)

    ################
    
    parser.add_argument("--encoder_lr_divider", help="how much to divide lr for encoder", type=float, default=100.0)  

    #HARDWARE###################################################################

    parser.add_argument("--gpus", help="number of gpus or gpus list, -1 uses all GPUs. use -1 for multinode, if want to specify which GPUs to use specify as comma seperated str with brackets e.g. [0, 1]", default="-1")
    
    parser.add_argument("--distributed_strategy", help="distributed strategy - ddp_spawn, ddp, fsdp_native, or None", default='ddp')


    #TRAINING#########################################################

    parser.add_argument("--peak_learning_rate", help="peak learning rate after warm up", type=float, default=0.02)

    parser.add_argument("--batch_size_per_device", help="batch size (PER DEVICE!!!, not effective). effective_batch_size is num_gpus * batch_size_per_device * accumulate_grad_batches", type=int, default=2)

    parser.add_argument("--accumulate_grad_batches", help="number of batches to accumulate before stepping optimizer. this is now computed automatically from effective_batch_size and cannot be set manually", type=int, default=-1)
    
    parser.add_argument("--effective_batch_size", help="desired effective batch size. accumulate_grad_batches is computed as effective_bs / (bs_per_device * num_gpus)", type=int, required=True)

    parser.add_argument("--gradient_clip_val", help="maximum value for gradient clipping", type=float, default=1.0)
    
    parser.add_argument("--is_random_seed", help="is_random_seed", action="store_true", default=False)
    
    parser.add_argument("--deterministic", help="ensures results are determnistic, may run a bit slower, sets flag on pl trainer and sets workers for dataloader", action="store_true", default=False)

    parser.add_argument("--execution_mode", type=str, choices=["pretrain", "finetune", "inference"], default="pretrain")
    
    parser.add_argument("--finetuning_model_ckpt", help="model ckpt when finetuning", type=str, default=None)

    #METRICS#########################################################
    #NOTE reported metrics in wandb will be averages over the time span they are being computed, so the final one in an epoch will be the most accurate

    parser.add_argument("--metrics_list", help = "list of metrics to use", nargs='+', default = [])
    
    parser.add_argument("--metrics_average_type", help="type of average to use for accuracy - macro, micro, weighted - see torchmetrics docs for more info", type=str, default="micro")

    parser.add_argument("--num_classes", help="the number of classes, must set if using metrics, see torchmetrics docs for more info", type=int, default=-1)

    parser.add_argument("--metrics_task", help="the type of task being done ['binary', 'multiclass' or 'multilabel'], see torchmetrics docs for more info, must set if using metrics_list", type=str, default="")

    parser.add_argument("--run_online_evaluation", help="run online evaluation", action="store_true", default=False)
    
    #OPTIMIZER AND LR SCHEDULER#########################################################
    
    parser.add_argument("--weight_decay", help="weight decay to use", type=float, default=0.01)
    
    parser.add_argument("--beta1", help="exponential decay rate for first moment estimate", type=float, default=0.9)
    
    parser.add_argument("--beta2", help="exponential decay rate for second movement estimate", type=float, default=0.999)

    parser.add_argument("--lr_scaling_rule", help="the LR will be scaled according to the rule LR = base_lr * effective_batch_size / 256. is useful for prototyping and is popular in visual SSL", action="store_true", default=False)

    parser.add_argument("--min_lr_scale", help="the most the lr will be scaled down during cosine decay", type=int, default=10)

    parser.add_argument("--max_steps", help="max number of steps for training", type=int, default=1000000)

    parser.add_argument("--max_scheduling_steps", help="max number of steps used for lr/other hparam scheduling. in general should be the same as max_steps but can be different if dont want to do a full run but want the lr and other things to be scheduled the same. similar to V jepa paper. if is not set will default to max_steps value", type=int, default=-1)
        
    parser.add_argument("--warm_up_steps", help="number of steps to increase the LR linearly over before hitting specified peak LR, starts from 0 unless warm_up_base_lr_divider is set and does linear warm up", type=int, default=10000)
    
    parser.add_argument("--warm_up_base_lr_divider", help="lr divider for when doing linear learning rate warm up. if is set to -1 then does warm up from 0", type=float, default=-1)
    
    parser.add_argument("--optimizer", help="optimizer to use, only adamw is supported", choices=["adamw"], type=str, default="adamw")

    #DATASET AND DATALOADER #########################################################

    parser.add_argument("--num_workers_per_device", help="num_workers per GPU. idea to do per GPU gotten from https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/", type=int, default=4)

    parser.add_argument("--prefetch_factor", help="prefetch factor for dataloader", type=int, default=None)
    
    parser.add_argument("--dataset_name", help="dataset name", default="something")
    
    parser.add_argument("--dataset_dir", help="dataset base directory", default="")
    
    parser.add_argument("--image_dims", help="List of image dimensions", nargs='+', type=int, default = [224, 224])
    
    parser.add_argument("--no_randomness_dataloader", help="makes dataloader have no randomness by only sampling from start", action="store_true", default=False)
        
    parser.add_argument("--preprocess_data", help="center crops images - helps to avoid black bars on sides which cause issues with unstable gradient. not used anymore", action="store_true", default=False)
    
    parser.add_argument("--crop_all_samples", help="if preprocess data is enabled this will crop all samples. not used anymore", action="store_true", default=False)
    
    parser.add_argument("--custom_image_normalization", help="whether to normalize data according to each dataset's std and mean", action="store_true", default=False)

    parser.add_argument("--ffprobe_path", help="path to ffprobe binary", default="")

    parser.add_argument("--validation_split_pct", help="if training and validation set are being split - which percent is validation", type=float, default=0.1)
    
    parser.add_argument("--test_split_pct", help="if training, validation, and test set are being split - which percent is test. by default this is not used", type=float, default=0)
    
    parser.add_argument("--limit_train_batches", help="percent of training dataset to use, or if > 1 is num batches", type=float, default=1.0)
    
    parser.add_argument("--limit_val_batches", help="percent of validation dataset to use, or if > 1 is num batches", type=float, default=1.0)
    
    parser.add_argument("--limit_test_batches", help="percent of testing dataset to use, or if > 1 is num batches", type=float, default=1.0)
    
    parser.add_argument("--val_sanity", help="number of sanity validation steps", type=int, default=0)
    
    parser.add_argument("--val_check_interval", help="interval to do validation per training epoch, 1 means once per epoch. useful if epochs are too large to wait to do validation", type=float, default=1.0)

    parser.add_argument("--check_val_every_n_epoch", help="how many epochs to wait before doing validation. is useful when you have a small dataset and dont want to do validation every time", type=int, default=1)

    #TESTING / INFERENCE#########################################################

    parser.add_argument("--run_testing_after_training", help="evaluate on test dataset. by default is off since repo is mostly pre-training, and may not work :/", action="store_true", default=False)

    parser.add_argument("--only_test", help="just testing no training, used passed in model checkpoint", action="store_true", default=False)

    parser.add_argument("--only_test_model_ckpt", help="model ckpt when only testing", type=str, default=None)
    
    #WANDB##################################################################

    parser.add_argument("--no_wandb", help="no wandb", action="store_true", default=False)

    parser.add_argument("--wandb_entity", help="wandb entity", default='')

    parser.add_argument("--wandb_project", help="wandb project name", default='XM')

    parser.add_argument("--wandb_tags", help="wandb tags to add", nargs='+', default=None)

    parser.add_argument("--wandb_offline", help="set wandb to offline mode", action="store_true", default=False)

    parser.add_argument("--wandb_watch", help="turns on watch mode for wandb - expensive so only use for debugging", action="store_true", default=False) 

    parser.add_argument("--wandb_watch_log_freq", help="number of steps to log for wandb watch. is higher since is a bit expensive", type = int, default=5000)  

    #LOGGING##################################################################

    parser.add_argument("--debug_log_filename", help="filename of log, deprecated", default='debug.log')

    parser.add_argument("--console_log_filename", help="filename of log, used when no_wandb is active", default='console.log')

    parser.add_argument("--print_logs", help="[deprecated] print to console, default false", action="store_true", default=False)

    parser.add_argument("--log_model_archi", help="log model architecture", action="store_true", default=False)

    parser.add_argument("--log_gradients", help="logs gradients at every step to wandb to debug them", action="store_true", default=False)

    parser.add_argument("--log_every_n_steps", help="turns on logger freq via pl, not advised to use this", type=int, default=50)

    # CHECKPOINTING ##################################################################

    parser.add_argument("--resume_training_ckpt", help="checkpoint to resume training from, use absolute",type=str, default="")     

    parser.add_argument("--checkpoint_monitor_string", help="string to use to monitor for saving checkpoint. supported by PL callback", type=str, default="valid_loss")

    parser.add_argument("--checkpoint_monitor_mode", help="monitoring mode for checkpoint_monitor_string, either ['min', 'max']. if is loss do min, if is a metric like accuracy do max", type=str, default="min")

    parser.add_argument("--save_top_k_ckpts", help="number of ckpts to save when doing val (saves the ones with best metrics using checkpoint monitor string and mode defined). -1 means save all", type=int, default=2)

    parser.add_argument("--checkpoint_base_dir", help="base directory for checkpoints, run name and timestamp are appended", type=str, default="./logs/checkpoints")

    #PRECISION#########################################################################

    parser.add_argument("--set_matmul_precision", help="set math mult precision - \"medium\", \"high\", or \"highest\" ", default=None)

    parser.add_argument("--float_precision", help="float precision, pl recommends 16-mixed/bf16-mixed, also has by default 32-true", type=str, default="32-true")

    # SPEED ##################################################################

    parser.add_argument("--compile_model", help="compiles the model using torch.compile", action="store_true", default=False)

    #SLURM#########################################################################

    parser.add_argument("--is_slurm_run", help="please set to true if doing slurm run, as of now just stops capturing console logs", action="store_true", default=False)
    
    parser.add_argument("--override_slurm_checks", help="dont use slurm checks to assert that certain conditions are true (i.e. train/test limit_batches is default value)", action="store_true", default=False)
    
    #DEBUGGING#######################################################################

    parser.add_argument("--debug_mode", help="turns debug mode on where dataset returned is very small, no_wandb is on and detect anomaly is on", action="store_true", default=False)

    parser.add_argument("--fast_dev_run", help="turns fast_dev_run for trainer on, makes it just do one training epoch and one val epoch", action="store_true", default=False)

    parser.add_argument("--debug_dataloader", help="makes mode to just debug dataloader", action="store_true", default=False)

    parser.add_argument("--overfit_batches", help="if nonzero will overfit to specified num/percent of batches", type=float, default=0.0)

    parser.add_argument("--profiler", choices=["simple", "advanced"], type=str, default="")

    parser.add_argument("--no_shuffle", help="stops shuffling - helpful for debugging", action="store_true", default=False)

    parser.add_argument("--detect_anomaly", help="turns on anomaly detection mode", action="store_true", default=False)

    parser.add_argument("--find_unused_parameters", help="turns on pl find unused params mode - DO NOT KEEP ON for actual training, helpful if want to debug. this uses DDPStrategy and ignores distributed_strategy", action="store_true", default=False)

    parser.add_argument("--debug_unused_parameters", help="makes it so it tracks which params are used to find the params that are causing the unused params issue. need to do some things in base_model_trainer so ctrl f this hparam to see the NOTEs", action="store_true", default=False)

    parser.add_argument("--manual_gc_collect_every_n_steps", help="manually call gc collect every n steps, can be done to prevent CPU RAM memory 'leak'", type = int, default=-1)

    parser.add_argument("--custom_limit_train_batches", help="if < 1, fraction of train samples to use; if >= 1, absolute number of train samples", type=float, default=0.0)


    ########################################################################

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)