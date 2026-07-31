### COMPUTE INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
# #SBATCH --exclusive

### LOG INFO ###
#SBATCH --job-name=baseline_transformer-lm
#SBATCH --output=logs/slurm/nlp/%x_%A-%a.log
mkdir -p logs/slurm/nlp/
command -v module &>/dev/null && module purge



${SLURM_ARRAY_TASK_ID:+srun } python train_model.py \
--model_name "baseline_transformer" \
--model_size "small" \
--run_name_hparams "xm_k=xm_best_of_k,prec=float_precision" \
--modality "NLP" \
\
--pretokenize_dataset \
--tokenizer "EleutherAI/gpt-neox-20b" \
--run_online_evaluation \
--run_online_lm_eval_every_n_val 1 \
--online_lm_eval_tasks "arc_easy,hellaswag,piqa,openbookqa,sciq,lambada_openai,wikitext" \
\
--context_length 256 \
\
--gpus "-1" \
\
--peak_learning_rate 0.0002 \
--batch_size_per_device 64 \
--effective_batch_size 64 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.1 \
--min_lr_scale 10 \
--max_steps 500000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name "fineweb" \
--num_workers_per_device 12 \
--validation_split_pct 0.0005 \
--val_check_interval 15000 \
\
--wandb_project 'nlp_pretrain' \
\
--save_top_k_ckpts 3 \
--log_model_archi \
--log_gradients \
\
--set_matmul_precision "medium" \
--float_precision "bf16-mixed" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}