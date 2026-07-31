# This module implements an lm-evaluation-harness model adapter and is structurally
# derived from EleutherAI's lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness),
# MIT License, Copyright (c) 2020 EleutherAI. The TemplateLM interface, the
# loglikelihood/generate_until request handling, and the batching and rolling-window
# logic follow the upstream HFLM implementation, adapted here for this codebase's models.
from lm_eval.api.model import TemplateLM


from typing import List, Tuple, TypedDict, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pytorch_lightning import seed_everything


import sys
from pathlib import Path

wd = Path(__file__).parent.parent.parent.resolve() # so can run and use files in main dir
sys.path.append(str(wd))

from tqdm import tqdm
import json
import os

import warnings
import datasets

from model.model_utils import load_trained_pl_model

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

# NOTE this code is deterministic with temp 0


class LocalLMWrapper(TemplateLM):

    def __init__(
        self,
        ckpt_path: str = None,
        tokenizer: str = "EleutherAI/gpt-neox-20b",
        device=torch.device('cuda'),
        batch_size: int = 32,
        loaded_model=None,
        loaded_hparams=None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if loaded_model is not None:
             self.model = loaded_model
             self.hparams = loaded_hparams
        else:
            override_pretrained_params = {} #NOTE can override pretrained params here if added them later
            
            self.model, self.hparams = load_trained_pl_model(ckpt_path, for_inference=True, return_pretrained_hparams=True, override_pretrained_params=override_pretrained_params)
            self.model = self.model.to(device=device)
            self.model.eval()

        self.max_gen_toks = 32

        self.batch_size = batch_size
        self.max_length = self.hparams['context_length']
        self.device=device

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(string, **kwargs, padding=True)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        """
        Computes log-likelihoods for a batch of context and continuation token sequences.

        Args:
            requests: List of tuples containing (context_str, continuation_str), context_tokens, continuation_tokens
            disable_tqdm: Whether to disable the progress bar

        Returns:
            List of tuples containing (log_likelihood, is_greedy) for each request
        """
        # Use provided batch size or override
        batch_size = self.batch_size
        results = []
        is_mdlm = hasattr(self.model, 'forward_lm_eval')

        # Process requests in batches
        for i in tqdm(range(0, len(requests), batch_size), disable=disable_tqdm, desc="Computing loglikelihoods"):
            # for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]
            batch_inps = []
            batch_cont_toks = []
            batch_inplens = []

            # Prepare inputs for each request in batch
            for _, context_enc, continuation_enc in batch:
                if is_mdlm:
                    # MDLM: include full continuation (logit at pos i predicts token at pos i, no next-token shift)
                    inp = torch.tensor(
                        ([self.tokenizer.bos_token_id] + context_enc + continuation_enc)[-self.max_length :],
                        dtype=torch.long,
                        device=self.device,
                    )
                else:
                    # AR: drop last token (logit at pos i predicts token at pos i+1)
                    inp = torch.tensor(
                        ([self.tokenizer.bos_token_id] + context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self.device,
                    )
                inplen = inp.shape[0]

                batch_inps.append(inp)
                batch_cont_toks.append(continuation_enc)
                batch_inplens.append(inplen)

            # Pad sequences in batch to same length
            padding_length = max(len(inp) for inp in batch_inps)
            padded_inps = []

            for inp in batch_inps:
                if len(inp) < padding_length:
                    # Right pad with pad token
                    padded_inp = torch.cat(
                        [inp, torch.full((padding_length - len(inp),), self.tokenizer.pad_token_id, device=self.device)]
                    )
                    padded_inps.append(padded_inp)
                else:
                    padded_inps.append(inp)

            batch_inps = torch.stack(padded_inps)

            # Get model predictions
            with torch.no_grad():
                if is_mdlm:
                    # mask continuation positions, keep context unmasked
                    mask_positions = torch.zeros_like(batch_inps, dtype=torch.bool)
                    for j, (inplen, cont_toks) in enumerate(zip(batch_inplens, batch_cont_toks)):
                        cont_len = len(cont_toks)
                        mask_positions[j, inplen - cont_len : inplen] = True
                    logits = self.model.forward_lm_eval(batch_inps, mask_positions)
                else:
                    logits = self.model(batch_inps, return_raw_logits=True, learning=False)

            # Convert to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Calculate result for each sample in batch
            for log_prob_seq, inplen, cont_toks in zip(log_probs, batch_inplens, batch_cont_toks):
                # Get logits for continuation tokens only
                cont_len = len(cont_toks)
                relevant_logits = log_prob_seq[inplen - cont_len : inplen]

                # Get log probs for the actual continuation tokens
                cont_token_tensors = torch.tensor(cont_toks, device=self.device)
                cont_log_probs = torch.gather(relevant_logits, 1, cont_token_tensors.unsqueeze(1)).squeeze(1)

                # Sum log probs
                total_log_prob = cont_log_probs.sum().item()

                # Check if continuation matches greedy prediction
                pred_tokens = relevant_logits.argmax(dim=1)
                is_greedy = (pred_tokens == cont_token_tensors).all().item()

                results.append((total_log_prob, is_greedy))
        print("Finished computing loglikelihoods.")
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        """
        Computes rolling log-likelihoods for a list of text strings.

        Args:
            requests: List of text requests
            disable_tqdm: Whether to disable the progress bar

        Returns:
            List of log-likelihoods for each request
        """
        loglikelihoods = []

        for (string,) in tqdm(
            [req.args for req in requests], disable=disable_tqdm, desc="Computing rolling loglikelihoods"
        ):
            # Get token windows with context length of 1
            token_list = self.tok_encode(string)
            windows = []

            for i in range(1, len(token_list)):
                # Context is just the previous token
                context = token_list[i - 1 : i]
                # Continuation is the current token
                continuation = token_list[i : i + 1]
                windows.append((("", ""), context, continuation))

            # Get log-likelihood for each window
            window_lls = self._loglikelihood_tokens(windows, disable_tqdm=True)

            # Sum the log-likelihoods (discarding is_greedy)
            total_ll = sum(ll for ll, _ in window_lls)
            loglikelihoods.append(total_ll)

        return loglikelihoods

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        """
        Generates text for each request until EOS token is reached.
        Uses simple greedy decoding. For MDLM, uses single-pass masked prediction.

        Args:
            requests: List of instances containing generation requests
            disable_tqdm: Whether to disable progress bar

        Returns:
            List of generated text strings
        """
        results = []
        is_mdlm = hasattr(self.model, 'forward_lm_eval')

        for request in tqdm(requests, disable=disable_tqdm, desc="Generating responses"):
            context, gen_kwargs = request.args

            # Extract stop sequences if any
            kwargs = gen_kwargs.copy() if isinstance(gen_kwargs, dict) else {}
            until = kwargs.pop("until", None)
            stop_sequences = []
            if until:
                if isinstance(until, str):
                    stop_sequences = [until]
                elif isinstance(until, list):
                    stop_sequences = until

            # Get max tokens to generate
            max_new_tokens = kwargs.pop("max_gen_toks", self.max_gen_toks)

            # Encode context
            context_tokens = self.tok_encode(context)
            context_tensor = torch.tensor(context_tokens, dtype=torch.long, device=self.device).unsqueeze(0)

            # Generate tokens
            with torch.no_grad():
                if is_mdlm:
                    ctx_len = context_tensor.shape[1]
                    gen_len = min(max_new_tokens, self.max_length - ctx_len)
                    if gen_len <= 0:
                        results.append("")
                        continue

                    if self.hparams.infer_mdlm_ar:
                        # AR-style: generate one token at a time, left to right
                        mask_pad = torch.full((1, gen_len), self.tokenizer.pad_token_id, device=self.device, dtype=torch.long)
                        full_input = torch.cat([context_tensor, mask_pad], dim=1)[:, -self.max_length:]
                        output_ids = []
                        for pos in range(ctx_len, ctx_len + gen_len):
                            mask_positions = torch.zeros_like(full_input, dtype=torch.bool)
                            mask_positions[:, pos:ctx_len + gen_len] = True
                            logits = self.model._forward_lm_eval_single(full_input, mask_positions)
                            next_token = logits[:, pos, :].argmax(dim=-1).item()
                            output_ids.append(next_token)
                            full_input[:, pos] = next_token
                            if next_token == self.eot_token_id:
                                break
                            if stop_sequences:
                                generated_text = self.tok_decode(output_ids)
                                if any(seq in generated_text for seq in stop_sequences):
                                    break
                    else:
                        # Single-pass: append mask tokens, predict all at once
                        mask_pad = torch.full((1, gen_len), self.tokenizer.pad_token_id, device=self.device, dtype=torch.long)
                        full_input = torch.cat([context_tensor, mask_pad], dim=1)[:, -self.max_length:]
                        mask_positions = torch.zeros_like(full_input, dtype=torch.bool)
                        mask_positions[:, ctx_len:] = True
                        logits = self.model.forward_lm_eval(full_input, mask_positions)
                        output_ids = logits[:, ctx_len:, :].argmax(dim=-1).squeeze(0).tolist()
                else:
                    # AR: autoregressive token-by-token generation
                    output_ids = []
                    current_input = context_tensor

                    for _ in range(max_new_tokens):
                        with torch.no_grad():
                            logits = self.model(current_input, return_raw_logits=True, learning=False)
                        next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
                        next_token_id = next_token.item()
                        output_ids.append(next_token_id)

                        if next_token_id == self.eot_token_id:
                            break

                        if stop_sequences:
                            generated_text = self.tok_decode(output_ids)
                            if any(seq in generated_text for seq in stop_sequences):
                                break

                        current_input = torch.cat([current_input, next_token], dim=-1)
                        if current_input.size(1) > self.max_length:
                            current_input = current_input[:, -self.max_length :]

                # Decode the generated tokens
                generated_text = self.tok_decode(output_ids)

                # Apply stop sequence trimming if needed
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in generated_text:
                            generated_text = generated_text.split(stop_seq)[0]
                            break

                results.append(generated_text)

        return results


def prepare_results(results, save_filepath, print_results=True):
    from lm_eval.utils import make_table

    if print_results:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

    json_result = json.dumps(results, indent=2, ensure_ascii=False, default=str)
    save_filepath.parent.mkdir(parents=True, exist_ok=True)
    save_filepath.open("w", encoding="utf-8").write(json_result)
    # return make_table(results, "groups")


def quick_eval_check(
    ckpt_path="",
    tokenizer="EleutherAI/gpt-neox-20b",
    device=torch.device("cuda"),
    out_dir="logs/inference/lm_eval",
    tasks=None,
    num_fewshot=0,
    batch_size=16,
    limit=None,
    seed=33,
    wandb_entity="",
    wandb_project="lm_eval",
    wandb_name="",
):  
    assert ckpt_path != "", "must provide checkpoint path"

    print(f"\nRunning evaluation on tasks: {tasks}")
    print(f"\nTrained Checkpoint file path: {ckpt_path}")

    print(f"run_name: {wandb_name}")

    seed_everything(seed, workers=True) #33 is default
    # print gpus available
    print(f"Number of Available GPUs: {torch.cuda.device_count()}")
    # print which gpus are available

    print(f"Available GPU IDs: {torch.cuda.current_device()}")
    if tasks is None:
        from lm_eval.tasks import TaskManager

        taskm = TaskManager()
        print("\n".join(taskm.task_index.keys()))
        print(
            "\n\nTo evaluate multiple tasks, you can chain the task names "
            "listed above via a comma-separated list."
            "\nFor example: `--tasks 'hellaswag,truthfulqa_mc2,mmlu'`. "
            "\nTo search for a specific task, use `recpre evaluate list | grep task_name`."
        )
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = "/".join(ckpt_path.split("/")[-2:]).strip(".ckpt")
    print("\n\nmodel_name", model_name)
    # base_dir = out_dir / model_name
    save_filepath = out_dir / Path(f"{model_name}.json")
    print("\n\nsave_filepath", save_filepath)

    from lm_eval import evaluator

    model = LocalLMWrapper(ckpt_path, tokenizer, device=device, batch_size=batch_size)

    print("Finished instantiating model, running evaluation now\n\n")

    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks.split(","),
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=str(device),
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )
    prepare_results(results, save_filepath)

    print("Finished running evaluation, logging stuff now\n\n")

    # log results to wandb
    from numbers import Number
    result_dict = {}

    def add_numeric(prefix, obj):
        if isinstance(obj, Number):
            result_dict[prefix] = round(float(obj), 4)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                name = f"{prefix}/{k}" if prefix else k
                add_numeric(name, v)

    for task_name, metrics in results.get("results", {}).items():
        add_numeric(task_name, metrics)

    # also log group metrics if present
    for group_name, group_metrics in results.get("groups", {}).items():
        add_numeric(f"groups/{group_name}", group_metrics)

    import wandb

    wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_name, dir=out_dir)
    if result_dict:
        wandb.log(result_dict)
    print("================================================================================")
    print(f"\n\nresult_dict: {result_dict}")
    print("================================================================================")


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Run lm-eval on a local checkpoint.")

    parser.add_argument("--ckpt_path", required=True, help="Path to the PyTorch Lightning checkpoint.")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Device for evaluation.")
    parser.add_argument("--out_dir", default="logs/inference/lm_eval", help="Directory to store results.")
    parser.add_argument("--tasks", default="arc_easy,hellaswag,piqa,openbookqa,sciq", help="Comma separated list of lm-eval tasks.")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (debugging).")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--wandb_entity", default="", help="Weights & Biases entity.")
    parser.add_argument("--wandb_project", default="lm_eval", help="Weights & Biases project.")
    parser.add_argument("--wandb_name", default=None, type=str, help="Run name for Weights & Biases.")

    args = parser.parse_args()

    quick_eval_check(
        ckpt_path=args.ckpt_path,
        tokenizer=args.tokenizer,
        device=torch.device(args.device),
        out_dir=args.out_dir,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        seed=args.seed,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )