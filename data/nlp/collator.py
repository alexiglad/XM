from transformers import AutoTokenizer, DataCollatorWithPadding

class NLP_HF_Collator:
    def __init__(self, hparams):
        self.hparams = hparams
        self.max_length = hparams.context_length+1
        self.tokenizer = None  # Will be initialized in __call__
        self.data_collator = None

    def __call__(self, batch):
        padding = True
        if self.hparams.pretokenize_dataset:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces=False)
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id # is token 0, was right padding things
                self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt", padding=padding, max_length=self.max_length)
            return self.data_collator(batch)
        else:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces = False)
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id # is token 0, was right padding things
            if self.hparams.execution_mode == "inference":
                questions, answers = zip(*batch)
                return self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length), self.tokenizer(answers, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            
            tokens = self.tokenizer(batch, return_tensors="pt", padding=padding, truncation=True, max_length=self.max_length)
            return tokens
