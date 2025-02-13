import os
from typing import Iterator
import itertools
import argparse
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import MarianTokenizer, MarianConfig, MarianMTModel
from torch.nn.modules.dropout import _DropoutNd
from nltk.translate.bleu_score import corpus_bleu

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def no_reproducibility() -> None:
    torch.backends.cudnn.benchmark = False

def shift_tokens(ids: torch.Tensor, op: str, tokenizer: MarianTokenizer) -> torch.Tensor:
    if op == "+":
        ids[
            (ids != tokenizer.bos_token_id) &
            (ids != tokenizer.pad_token_id) &
            (ids != tokenizer.eos_token_id) &
            (ids != tokenizer.mask_token_id) &
            (ids != tokenizer.unk_token_id)
        ] += len(tokenizer)
    elif op == "-":
        ids[
            (ids >= len(tokenizer))
        ] -= len(tokenizer)
    return ids

class BasicDataset(IterableDataset):
    def __init__(self, X_path: str, Y_path: str, parallel: bool):
        self.X_path = X_path
        self.Y_path = Y_path
        self.parallel = parallel
        
    def __iter__(self) -> Iterator[tuple[str, str]]:
        X_file = open(self.X_path, "r", encoding="utf-8", errors="replace")
        Y_file = open(self.Y_path, "r", encoding="utf-8", errors="replace")
        if self.parallel:
            it = zip(X_file, Y_file)
        else:
            it = zip(itertools.cycle(X_file), itertools.cycle(Y_file))
        return it

class CollateWithTokenizer():
    def __init__(self, tokenizer_X: MarianTokenizer, tokenizer_Y: MarianTokenizer, max_length: int, device: str):
        self.tokenizer_X = tokenizer_X
        self.tokenizer_Y = tokenizer_Y
        self.max_length = max_length
        self.device = device

    def __call__(self, batch: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized_X = self.tokenizer_X([X[0] for X in batch], return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        tokenized_Y = self.tokenizer_Y([Y[1] for Y in batch], return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
                                     
        X_ids = tokenized_X["input_ids"].to(self.device)
        X_masks = tokenized_X["attention_mask"].to(self.device)
        Y_ids = tokenized_Y["input_ids"].to(self.device)
        Y_masks = tokenized_Y["attention_mask"].to(self.device)
    
        Y_ids = shift_tokens(Y_ids, "+", self.tokenizer_Y)
        
        return X_ids, X_masks, Y_ids, Y_masks

class WordDropout(_DropoutNd):
    # https://gist.github.com/JohnGiorgi/c030de1dd8cb84ad0970d1cc87e2ed86
    def __init__(self, p: float, eos_token_id: int, pad_token_id: int):
        super(WordDropout, self).__init__(p)
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def forward(self, input: torch.Tensor, mask_token_id: int) -> torch.Tensor:
        if not self.training or not self.p:
            return input

        keep = torch.empty_like(input).bernoulli_(1 - self.p).bool()
        eos = (input == self.eos_token_id)
        pad = (input == self.pad_token_id)
        keep_eos = keep + eos + pad
        input = torch.where(keep_eos, input, torch.empty_like(input).fill_(mask_token_id))
        return input

def get_config(params: argparse.Namespace) -> MarianConfig:
    return MarianConfig(
        vocab_size=params.vocab_size,
        d_model=params.d_model,
        encoder_layers=params.n_encoder,
        decoder_layers=params.n_decoder,
        encoder_attention_heads=params.n_encoder_head,
        decoder_attention_heads=params.n_decoder_head,
        decoder_ffn_dim=params.d_encoder,
        encoder_ffn_dim=params.d_decoder,
        activation_function="relu",
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        max_position_embeddings=params.max_length,
        init_std=0.02,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        scale_embedding=True,
        use_cache=True,
        pad_token_id=1,
        bad_words_ids=[[1]],
        bos_token_id=0,
        eos_token_id=2,
        forced_eos_token_id=2,
        max_length=params.max_length,
        normalize_embedding=False,
        share_encoder_decoder_embeddings=True,
        is_encoder_decoder=True,
        static_position_embeddings=True,
        torch_dtype="float32",
        decoder_start_token_id=1,
        decoder_vocab_size=params.vocab_size,
    )

def calculate_bleu_score(reference_path: str, hypothesis_path: str) -> float:
    with open(reference_path, "r", encoding="utf-8") as ref_file:
        references = [[line.strip().split() for line in ref_file.readlines()]]
    
    with open(hypothesis_path, "r", encoding="utf-8") as hyp_file:
        hypotheses = [line.strip().split() for line in hyp_file.readlines()]
    
    return corpus_bleu(references, hypotheses)

def translate_validation_set(i: int, G: MarianMTModel, F: MarianMTModel, tokenizer_X: MarianTokenizer, tokenizer_Y: MarianTokenizer, max_length: int, validation_dataloader: DataLoader) -> None:
    G = G.eval()
    F = F.eval()
    X2Y_translations = []
    Y2X_translations = []
    with torch.no_grad():
        for X_ids, X_masks, Y_ids, Y_masks in validation_dataloader:
            fake_Y = G.generate(X_ids, max_new_tokens=max_length)
            fake_Y = shift_tokens(fake_Y, "-", tokenizer_Y)
            fake_X = F.generate(Y_ids, max_new_tokens=max_length)
    
            X2Y_translations.extend(tokenizer_Y.batch_decode(fake_Y, skip_special_tokens=True))
            Y2X_translations.extend(tokenizer_X.batch_decode(fake_X, skip_special_tokens=True))
    
    with open(os.path.join("output", f"{i}_X2Y.txt"), "w+", encoding="utf-8") as f:
        for line in X2Y_translations:
            f.write(f"{line}\n")
    
    with open(os.path.join("output", f"{i}_Y2X.txt"), "w+", encoding="utf-8") as f:
        for line in Y2X_translations:
            f.write(f"{line}\n")
    
    G = G.train()
    F = F.train()