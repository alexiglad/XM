### COMPUTE INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
# #SBATCH --exclusive

### LOG INFO ###
#SBATCH --job-name=dit-cc
#SBATCH --output=logs/slurm/img/%x_%A-%a.log
mkdir -p logs/slurm/img/
command -v module &>/dev/null && module purge


${SLURM_ARRAY_TASK_ID:+srun } python train_model.py \
--model_name "dit" \
--model_size "vit_base" \
--run_name_hparams "img=image_dims,ps=patch_size,enc=backbone_type,prec=bf16m" \
--modality "IMG" \
\
--diffusion_supervision_type "velocity" \
--ode_step_size 0.02 \
\
--patch_size 2 \
--image_task "class_conditional" \
--num_classes 1000 \
--cfg_dropout_prob 0.1 \
--log_image_every_n_steps 10000 \
--check_val_every_n_epoch 5 \
--ema_model 0.9999 \
--run_online_evaluation \
--run_online_fid_eval_every_n_epochs 10 \
--online_fid_eval_num_samples 50000 \
--online_fid_cfg_scales 0.0 1.5 \
\
--gpus "-1" \
\
--peak_learning_rate 0.0001 \
--batch_size_per_device 256 \
--effective_batch_size 256 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 1000000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--backbone_type "vae" \
--use_cached_img_latents \
--cached_img_latents_dir "$HF_HOME/image_net_cache" \
--dataset_name "imagenet" \
--num_workers_per_device 12 \
--image_dims 256 256 \
\
--wandb_project "xm_img" \
\
--log_model_archi \
--log_gradients \
--log_every_n_steps 50 \
\
--set_matmul_precision "medium" \
--float_precision "bf16-mixed" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}