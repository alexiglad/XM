### COMPUTE INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
# #SBATCH --exclusive

### LOG INFO ###
#SBATCH --job-name=dit-wm
#SBATCH --output=logs/slurm/vid/%x_%A-%a.log
mkdir -p logs/slurm/vid/
command -v module &>/dev/null && module purge

best_of_k=(1 2 3 5 8 12 20)
# NOTE need to tune xm_chunk_bs_mult
# NOTE for initial preencoding need to set bs to be much smaller

${SLURM_ARRAY_TASK_ID:+srun } python train_model.py \
--model_name "dit" \
--model_size "vit_base" \
--run_name_hparams "img=image_dims,ps=patch_size,enc=backbone_type,xm_k=xm_best_of_k,inf_steps=fvd_inference_diffusion_steps" \
--modality "VID" \
--video_transformer_3d \
\
--xm_best_of_k ${best_of_k[${SLURM_ARRAY_TASK_ID}]} \
--xm_chunk_bs_mult 13 \
--xm_save_mem_mode \
\
--diffusion_supervision_type "velocity" \
--fvd_inference_diffusion_steps 50 \
\
--video_task "gc_world_model" \
--world_modeling_given_frames 0 1 9 \
--context_length 10 \
--log_video_every_n_steps 5000 \
--patch_size 4 \
--run_online_evaluation \
--run_online_fvd_eval_every_n_epochs 50 \
--online_fvd_eval_num_videos 10000 \
--ema_model 0.9999 \
--cfg_dropout_prob 0.1 \
--online_fvd_cfg_scales 0.0 1.5 \
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
--dataset_name "something" \
--dataset_dir ${SSV2_DIR} \
--num_workers_per_device 12 \
--image_dims 128 128 \
--preencode_dataset \
--hdf5_file_path "$HF_HOME/ssv2/vae_128x128.hdf5" \
\
--backbone_type "vae" \
--vae_normalization \
--time_between_frames 0.25 \
--sdxl_vae_standardization \
\
--wandb_project 'xm_wm' \
\
--log_model_archi \
--log_gradients \
--log_every_n_steps 50 \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}