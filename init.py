#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import json
import random 
import numpy as np
import torch
from torch.nn.modules.dropout import _DropoutNd
from transformers import MarianTokenizer, MarianConfig, MarianMTModel
import sentencepiece as spm
from torch.utils.data import IterableDataset, DataLoader
from cyclegn_utils import set_seeds, no_reproducibility, BasicDataset, CollateWithTokenizer, WordDropout, get_config

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", help="Random seed", type=int, default=4512540)
    parser.add_argument("--vocab_size", help="Size of the vocabulary", type=int, default=32000)
    parser.add_argument("--d_model", help="Dimension of the model", type=int, default=1024)
    parser.add_argument("--n_encoder", help="Number of encoder layers", type=int, default=6)
    parser.add_argument("--n_decoder", help="Number of decoder layers", type=int, default=6)
    parser.add_argument("--n_encoder_head", help="Number of encoder heads", type=int, default=8)
    parser.add_argument("--n_decoder_head", help="Number of decoder heads", type=int, default=8)
    parser.add_argument("--d_encoder", help="Dimension of the encoder layer", type=int, default=2048)
    parser.add_argument("--d_decoder", help="Dimension of the decoder layer", type=int, default=2048)
    parser.add_argument("--max_length", help="Maximum length (in tokens) of the sequences", type=int, default=128)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("--X_spm_train", help="X tokenizer dataset (optional)", type=str)
    parser.add_argument("--Y_spm_train", help="Y tokenizer dataset (optional)", type=str)
    parser.add_argument("--max_spm_samples", help="Maximum number of sequences during tokenizer training", type=int, default=100000)
    parser.add_argument("--X_train", help="X dataset", type=str)
    parser.add_argument("--Y_train", help="Y dataset", type=str)
    parser.add_argument("--max_train_samples", help="Maximum number of sequences", type=int, default=5000000)
    parser.add_argument("--no_reproducible", help="Do not enforce reproducibility (optional)", action=argparse.BooleanOptionalAction)
    return parser

def collate_fn(batch: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokenized_X = tokenizer_X([X[0] for X in batch], return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding="max_length")
    tokenized_Y = tokenizer_Y([Y[1] for Y in batch], return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding="max_length")
                                 
    X_ids = tokenized_X["input_ids"].to(device)
    X_masks = tokenized_X["attention_mask"].to(device)
    Y_ids = tokenized_Y["input_ids"].to(device)
    Y_masks = tokenized_Y["attention_mask"].to(device)

    Y_ids = shift_tokens(Y_ids, "+", tokenizer_Y)
    
    return X_ids, X_masks, Y_ids, Y_masks

def train_tokenizer(config: MarianConfig, dataset: str, name: str, max_spm_samples: int) -> None:
    spm.SentencePieceTrainer.train(
        input=dataset,
        model_prefix=name, 
        vocab_size=config.vocab_size//2, 
        pad_id=config.pad_token_id, 
        eos_id=config.eos_token_id, 
        unk_id=3, 
        bos_id=config.bos_token_id,
        control_symbols="<mask>",
        character_coverage=0.9995,
        model_type="unigram",
        input_sentence_size=max_spm_samples,
    )
    sp = spm.SentencePieceProcessor(model_file=f"{name}.model")
    vocab = {sp.IdToPiece(i): i for i in range(len(sp))}
    with open(f"vocab_{name}.json", "w+", encoding="utf-8") as f:
        json.dump(vocab, f)

def train(params: argparse.Namespace, config: MarianConfig, device: str) -> None:    
    tokenizer_X = MarianTokenizer(
        source_spm="X.model",
        target_spm="X.model",
        model_max_length=params.max_length,
        vocab="vocab_X.json",
    )
    tokenizer_X.mask_token = "<mask>"
    tokenizer_X.mask_token_id = tokenizer_X.get_vocab()[tokenizer_X.mask_token]
    
    tokenizer_Y = MarianTokenizer(
        source_spm="Y.model",
        target_spm="Y.model",
        model_max_length=params.max_length,
        vocab="vocab_Y.json",
    )
    tokenizer_Y.mask_token = "<mask>"
    tokenizer_Y.mask_token_id = tokenizer_Y.get_vocab()[tokenizer_Y.mask_token]

    collate_fn = CollateWithTokenizer(tokenizer_X, tokenizer_Y, params.max_length, device)
    training_dataset = BasicDataset(
        params.X_train,
        params.Y_train,
        False,
    )
    training_dataloader = DataLoader(
        training_dataset, 
        batch_size=params.batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
    )

    H = MarianMTModel(config)
    H = H.train()
    H = H.to(device)
    opt_H = torch.optim.AdamW(H.parameters(), lr=1e-4)
    
    word_dropout = WordDropout(0.15, config.eos_token_id, config.pad_token_id)

    x = []
    y = []
    for i, (X_ids, X_masks, Y_ids, Y_masks) in enumerate(training_dataloader, start=1):
        opt_H.zero_grad(set_to_none=True)
        
        partial_X_ids = word_dropout(X_ids, tokenizer_X.mask_token_id)
        partial_Y_ids = word_dropout(Y_ids, tokenizer_Y.mask_token_id)

        rec_X = H(input_ids=partial_X_ids, attention_mask=X_masks, labels=X_ids)
        rec_Y = H(input_ids=partial_Y_ids, attention_mask=Y_masks, labels=Y_ids)
            
        loss_total = rec_X.loss + rec_Y.loss
        x.append(rec_X.loss.item())
        y.append(rec_Y.loss.item())
        
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(H.parameters(), max_norm=2.0, norm_type=2)
        opt_H.step()

        if i%1000 == 0:
            print(f"ti: {i}\tAverage loss X: {np.array(x).mean()}\tAverage loss Y: {np.array(y).mean()}")
            x = []
            y = []

        if i%params.max_train_samples == 0:
            break

    torch.save(
        {
            "H": H.state_dict(),
            "opt_H": opt_H.state_dict(),
            "max_train_samples": params.max_train_samples,
        }, 
        "checkpoint_MLM_shared_shift.pth"
    )
    
def main(params: argparse.Namespace) -> None:
    set_seeds(params.seed)
    if params.no_reproducible:
        no_reproducibility()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = get_config(params)
    if params.X_spm_train:  
        train_tokenizer(config, params.X_spm_train, "X", params.max_spm_samples)
    else:
        train_tokenizer(config, params.X_train, "X", params.max_spm_samples)
    if params.Y_spm_train:  
        train_tokenizer(config, params.Y_spm_train, "Y", params.max_spm_samples)
    else:
        train_tokenizer(config, params.Y_train, "Y", params.max_spm_samples)
    train(params, config, device)

if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params)