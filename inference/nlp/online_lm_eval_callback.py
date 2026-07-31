import gc
from datetime import timedelta

import torch
from torchvision import transforms
from torchvision.transforms import ToPILImage
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
import wandb
from lm_eval import evaluator
from inference.nlp.lm_harness import LocalLMWrapper

from model.model_utils import *

# TODO eventually add multi GPU support for this
class LM_Eval_Callback(pl.Callback):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.validations_done = 0


    def on_fit_start(self, trainer, pl_module):
        """
        Called at the start of training.
        """
        # uncomment this for debugging where dont have to wait for lots of training
        # print(f"running online LM evaluation at epoch {trainer.current_epoch}")
        # self.run_online_lm_evaluation(trainer, pl_module, "valid")
        pass
    

    def on_validation_end(self, trainer, pl_module):
        """
        Called after every validation loop ends.
        """
        hparams = pl_module.hparams
        
        if ((self.validations_done % hparams.run_online_lm_eval_every_n_val)) == 0:
            print(f"running online LM evaluation at epoch {trainer.current_epoch}")
            with temporary_ddp_timeout(timedelta(minutes=120)): # rank 0 runs lm_eval (minutes) while other ranks wait at the trailing barrier
                self.run_online_lm_evaluation(trainer, pl_module, "valid")
                trainer.strategy.barrier()

        self.validations_done += 1
        

    def run_online_lm_evaluation(self, trainer, pl_module, phase):
        hparams = pl_module.hparams
        model = pl_module.model.eval()

        with torch.no_grad():
            if trainer.is_global_zero:
                tokenizer_name = hparams.tokenizer

                wrapper = LocalLMWrapper(
                    loaded_model=model,
                    loaded_hparams=hparams,
                    device=pl_module.device,
                    tokenizer=tokenizer_name
                )

                tasks = hparams.online_lm_eval_tasks
                if isinstance(tasks, str):
                    tasks = tasks.split(",")

                # avoid deadlock from tokenizer forking during bootstrap
                import os
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                print(f"Running LM evaluation on tasks: {tasks}")
                results = evaluator.simple_evaluate(
                    model=wrapper,
                    tasks=tasks,
                    batch_size=hparams.batch_size_per_device,
                    device=str(pl_module.device),
                    limit=32,
                    bootstrap_iters=0,
                )

                result_dict = {}
                acc_scores = []
                acc_norm_scores = []
                ppl_scores = []

                if "results" in results:
                    for task_name, metrics in results["results"].items():
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)):
                                result_dict[f"online_eval/{task_name}/{k}"] = v

                                if "acc" in k and "norm" not in k and "stderr" not in k:
                                    acc_scores.append(v)
                                elif "acc_norm" in k and "stderr" not in k:
                                    acc_norm_scores.append(v)
                                elif "perplexity" in k:
                                    ppl_scores.append(v)
                            elif isinstance(v, str): # sometimes metrics are strings? usually not for numbers we want to log
                                pass

                if "groups" in results:
                    for group_name, group_metrics in results["groups"].items():
                        for k, v in group_metrics.items():
                            if isinstance(v, (int, float)):
                                result_dict[f"online_eval/groups/{group_name}/{k}"] = v

                # Log averages
                if len(acc_scores) > 0:
                    result_dict["eval_ave_acc"] = sum(acc_scores) / len(acc_scores)
                if len(acc_norm_scores) > 0:
                    result_dict["online_eval/average_acc_norm"] = sum(acc_norm_scores) / len(acc_norm_scores)
                if len(ppl_scores) > 0:
                    result_dict["eval_ave_ppl"] = sum(ppl_scores) / len(ppl_scores)
                
                print(f"LM Eval Results: {result_dict}")
                if trainer.is_global_zero:
                    print("trainer.global_step", trainer.global_step)
                    trainer.logger.experiment.log({**result_dict, "trainer/global_step": trainer.global_step})

        torch.cuda.empty_cache()
        gc.collect()
