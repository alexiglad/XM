import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy

from transformers import AutoTokenizer

import math
import random
import os
from model.model_utils import *


class Baseline_Transformer_NLP(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces = False)
        self.tokenizer_pad_token_id = tokenizer.eos_token_id # is token 0, was right padding things
        
        self.vocab_size = len(tokenizer) # self.vocab_size = tokenizer.vocab_size caused errors since is smaller than len(tokenizer), is 50254 for neox-20b, len tokenizer is 50277 so decided to use that

        self.xm_lm_mode = self.hparams.xm_lm_embed_pct > 0
        if self.xm_lm_mode:
            assert self.hparams.xm_best_of_k > 1, "xm_lm_embed_pct > 0 requires xm_best_of_k > 1"
            assert self.hparams.xm_lm_embed_pct < 1, "xm_lm_embed_pct must be < 1 (fraction, not pct)"
            self.xm_dim = round(self.hparams.xm_lm_embed_pct * self.hparams.embedding_dim)
            assert self.xm_dim >= 1, "xm_lm_embed_pct is too small; resulting xm_dim rounded to 0"
            main_embed_dim = self.hparams.embedding_dim - self.xm_dim
            assert main_embed_dim >= 1, "xm_lm_embed_pct is too large; no room left for token embeddings"
            self.embeddings = nn.Embedding(self.vocab_size, main_embed_dim)
            self.xm_embeddings = nn.Embedding(self.hparams.xm_best_of_k, self.xm_dim)
            init_weights(self.xm_embeddings, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
        else:
            num_embeddings = self.vocab_size if self.hparams.xm_best_of_k <= 1 else self.vocab_size + self.hparams.xm_best_of_k
            self.embeddings = nn.Embedding(num_embeddings, self.hparams.embedding_dim)
        init_weights(self.embeddings, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)

        self.log_softmax = nn.LogSoftmax(dim = -1)

        self.transformer = setup_ar_transformer(self.hparams)
        
        self.output = nn.Linear(self.hparams.embedding_dim, self.vocab_size, bias = False)
        init_weights(self.output, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
        
        self.finished_warming_up = False
    

    def forward(self, x, learning = True, start_pos = 0, return_raw_logits = False, rand_seeds=None): # accepts input_ids as input
        if self.xm_lm_mode:
            # per-position Explorative Modeling concat mode: each token's embedding is [token_embed (main_dim) | xm_embed (xm_dim)]
            token_embeddings = self.embeddings(x) # BS, S, main_embed_dim
            B, S = x.shape
            if rand_seeds is None:
                xm_idx = torch.zeros(B, S, dtype=torch.long, device=x.device) # deterministic inference default
            else:
                # per-sample deterministic, per-position independent random best-of-k indices
                # (a cyclic (rand_seeds + pos) % K would make positions perfectly correlated within a sample)
                gen = torch.Generator(device='cpu')
                seed_list = rand_seeds.detach().cpu().tolist()
                xm_idx_cpu = torch.empty(B, S, dtype=torch.long)
                for b in range(B):
                    gen.manual_seed(seed_list[b])
                    xm_idx_cpu[b] = torch.randint(0, self.hparams.xm_best_of_k, (S,), generator=gen)
                xm_idx = xm_idx_cpu.to(x.device)
            xm_embeds = self.xm_embeddings(xm_idx) # BS, S, xm_dim
            embeddings = torch.cat([token_embeddings, xm_embeds], dim=-1) # BS, S, embedding_dim
        else:
            embeddings = self.embeddings(x) # x here is input_ids

            if self.hparams.xm_best_of_k > 1:
                if rand_seeds is None:
                    xm_indices = torch.randint(0, self.hparams.xm_best_of_k, (x.shape[0],), device=x.device)
                else:
                    xm_indices = rand_seeds % self.hparams.xm_best_of_k

                if start_pos == 0:
                    xm_input_ids = self.vocab_size + xm_indices
                    xm_embeddings = self.embeddings(xm_input_ids).unsqueeze(1) # BS, 1, D
                    embeddings = torch.cat([xm_embeddings, embeddings], dim=1)

        predicted_embeddings = self.transformer(embeddings, start_pos = start_pos, learning = learning) # BS, S, D
        predicted_logits = self.output(predicted_embeddings) #BS, S, vocab_size

        if (not self.xm_lm_mode) and self.hparams.xm_best_of_k > 1 and start_pos == 0: # remove the Explorative Modeling token
            predicted_logits = predicted_logits[:, 1:]

        if return_raw_logits:
            return predicted_logits
        else:
            predicted_distributions = self.log_softmax(predicted_logits).reshape(-1, self.vocab_size) # BS*S, V; reshape since preds for nll should be 2d
            return predicted_distributions


    def loss_calc_wrapper(self, model_forward, input_ids, gt_indices, learning=True, rand_inputs=None, rand_seeds=None):
        # NOTE this doesnt use rand_inputs
        with torch.set_grad_enabled(learning):
            predicted_distributions = model_forward(x=input_ids, learning=learning, rand_seeds=rand_seeds)
            gt_indices_reshaped = gt_indices.reshape(-1) # BS * S; reshape since targets are supposed to be 1D
            
            loss = F.nll_loss(predicted_distributions, gt_indices_reshaped, reduction="none", ignore_index=self.tokenizer_pad_token_id)
            loss = loss.reshape(input_ids.shape[0], -1)
            non_pad_mask = (gt_indices != self.tokenizer_pad_token_id).float()
            loss = (loss.sum(dim=1)) / non_pad_mask.sum(dim=1).clamp(min=1)

            return loss, None

        
    def forward_loss_wrapper(self, x, phase="train"):
        input_ids = x['input_ids'].squeeze(dim=1)[:, :-1] # x['input_ids'] shape is BS, S+1, only input first S--remove next tokens
        learning = (phase == "train")

        next_token_indices = x['input_ids'].squeeze(dim=1)[:, 1:] # B, S; squeeze was to remove 1 on 2nd dim

        xm_loss_comparable = None
        if self.hparams.xm_best_of_k <= 1:
            predicted_distribution = self(input_ids, learning=learning)

            next_token_indices = next_token_indices.reshape(-1) # BS * S; reshape since targets are supposed to be 1D
            cce_loss = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=self.tokenizer_pad_token_id)

        else: # Explorative Modeling
            cce_loss, __ = xm_chunked_best_of_k(self.forward, self.loss_calc_wrapper, input_ids, next_token_indices, self.hparams.xm_best_of_k, self.hparams.xm_chunk_bs_mult, save_mem_mode=self.hparams.xm_save_mem_mode, debug_save_mem_mode=self.hparams.xm_debug_mode, not_training=not learning)
            non_pad_counts = (next_token_indices != self.tokenizer_pad_token_id).sum(dim=1).float() # length-weighted mean to match F.nll_loss(reduction="mean")
            cce_loss = (cce_loss * non_pad_counts).sum() / non_pad_counts.sum()
            
            if not learning: # debug comparable val/test loss
                predicted_distribution = self(input_ids, learning)
                next_token_indices = next_token_indices.reshape(-1)
                xm_loss_comparable = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=self.tokenizer_pad_token_id)
        
        ppl_loss = torch.exp(cce_loss).detach()
        log_dict = {
            'loss': cce_loss,
            'perplexity': ppl_loss
        }

        if xm_loss_comparable is not None:
            log_dict['xm_loss_comparable'] = xm_loss_comparable

        return log_dict