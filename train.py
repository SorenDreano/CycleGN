#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import json
import random 
import numpy as np
import torch
from transformers import MarianTokenizer, MarianConfig, MarianMTModel
import sentencepiece as spm
from torch.utils.data import IterableDataset, DataLoader
from cyclegn_utils import set_seeds, no_reproducibility, BasicDataset, CollateWithTokenizer, get_config, calculate_bleu_score, translate_validation_set

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
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--single_embedding", help="Share a single embedding between G anf F (optional)", action=argparse.BooleanOptionalAction)
    parser.add_argument("--evaluate_set", help="Number of batches that run before the evaluation set is translated", type=int, default=1000)
    parser.add_argument("--X_train", help="X dataset", type=str)
    parser.add_argument("--Y_train", help="Y dataset", type=str)
    parser.add_argument("--X_valid", help="X dataset", type=str)
    parser.add_argument("--Y_valid", help="Y dataset", type=str)
    parser.add_argument("--max_train_samples", help="Maximum number of sequences", type=int, default=10000000)
    parser.add_argument("--no_reproducible", help="Do not enforce reproducibility (optional)", action=argparse.BooleanOptionalAction)
    return parser

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

    validation_dataset = BasicDataset(
        params.X_valid,
        params.Y_valid,
        True,
    )
    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size=params.batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
    )

    padder_fake = torch.full((params.batch_size, params.max_length), config.pad_token_id, dtype=torch.int64, requires_grad=False, device=device)
    criterion_cycle = torch.nn.CrossEntropyLoss()
    criterion_identity = torch.nn.CrossEntropyLoss()
    
    G = MarianMTModel(config)
    F = MarianMTModel(config)
    checkpoint = torch.load("checkpoint_MLM_shared_32000_shift.pth", map_location=device)
    G.load_state_dict(checkpoint["H"])
    F.load_state_dict(checkpoint["H"])
    del checkpoint
    if params.single_embedding:
        shared_embedding = G.model.shared
        F.model.shared = shared_embedding
        F.model.encoder.embed_tokens = shared_embedding
        F.model.decoder.embed_tokens = shared_embedding
    G = G.train()
    G = G.to(device)
    F = F.train()
    F = F.to(device)
    if params.single_embedding:
        parameters = {id(p): p for p in G.parameters()}
        parameters.update({id(p): p for p in F.parameters()})
        opt = torch.optim.AdamW(parameters.values(), lr=1e-4)
    else:
        opt_G = torch.optim.AdamW(G.parameters(), lr=1e-4)
        opt_F = torch.optim.AdamW(F.parameters(), lr=1e-4)
    

    x = []
    y = []
    z = []
    for i, (X_ids, X_masks, Y_ids, Y_masks) in enumerate(training_dataloader, start=1):
        if params.single_embedding:
            opt.zero_grad(set_to_none=True)
        else:
            opt_G.zero_grad(set_to_none=True)
            opt_F.zero_grad(set_to_none=True)
        G = G.eval()
        F = F.eval()
        
        with torch.no_grad():
            fake_Y = G.generate(X_ids, max_new_tokens=params.max_length)
            fake_X = F.generate(Y_ids, max_new_tokens=params.max_length)
            fake_X = torch.cat((fake_X, padder_fake), dim=1)[:, :params.max_length]
            fake_Y = torch.cat((fake_Y, padder_fake), dim=1)[:, :params.max_length]
    
        G = G.train()
        F = F.train()
    
        fake_X_masks = torch.full_like(fake_X, 0)
        fake_X_masks[fake_X >= config.eos_token_id] = 1
    
        fake_Y_masks = torch.full_like(fake_Y, 0)
        fake_Y_masks[fake_Y >= config.eos_token_id] = 1
    
        id_x = F(
            X_ids,
            attention_mask=X_masks,
            labels=X_ids
        ).logits
        id_y = G(
            Y_ids,
            attention_mask=Y_masks,
            labels=Y_ids
        ).logits
        
        rec_x = F(
            fake_Y,
            attention_mask=fake_Y_masks,
            labels=X_ids
        ).logits
        rec_y = G(
            fake_X,
            attention_mask=fake_X_masks,
            labels=Y_ids,
        ).logits        
        
        loss_cycle_XYX = criterion_cycle(rec_x.permute(0, 2, 1), X_ids)
        loss_cycle_YXY = criterion_cycle(rec_y.permute(0, 2, 1), Y_ids)
        loss_id_X = criterion_identity(id_x.permute(0, 2, 1), X_ids) * 0.1
        loss_id_Y = criterion_identity(id_y.permute(0, 2, 1), Y_ids) * 0.1
        loss_total = loss_cycle_XYX + loss_cycle_YXY + loss_id_X + loss_id_Y

        x.append(loss_cycle_YXY.item() + loss_id_Y.item())
        y.append(loss_cycle_XYX.item() + loss_id_X.item())
        z.append(loss_cycle_YXY.item() + loss_id_Y.item() + loss_cycle_XYX.item() + loss_id_X.item())
        
        loss_total.backward()

        if params.single_embedding:
            torch.nn.utils.clip_grad_norm_(parameters.values(), max_norm=2.0, norm_type=2)
            opt.step()
        else:
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=2.0, norm_type=2)
            torch.nn.utils.clip_grad_norm_(F.parameters(), max_norm=2.0, norm_type=2)
            opt_G.step()
            opt_F.step()

        if i%params.evaluate_set == 0: 
            translate_validation_set(i, G, F, tokenizer_X, tokenizer_Y, params.max_length, validation_dataloader)
            G_bleu = calculate_bleu_score(params.Y_valid, os.path.join("output", f"{i}_X2Y.txt"))
            F_bleu = calculate_bleu_score(params.X_valid, os.path.join("output", f"{i}_Y2X.txt"))
            print(f"i: {i}\tAverage loss: {np.array(z).mean()}\tAverage loss G: {np.array(x).mean()}\tAverage loss F: {np.array(y).mean()}\tG BLEU: {G_bleu}\tF BLEU: {F_bleu}")
            x = []
            y = []
            z = []

        if i%params.max_train_samples == 0:
            break

    if params.single_embedding:
        torch.save(
            {
                "G": G.state_dict(),
                "opt": opt.state_dict(),
                "F": F.state_dict(),
                "max_train_samples": params.max_train_samples,
            }, 
            "checkpoint_MLM_shared_shift.pth"
        )
    else:
        torch.save(
            {
                "G": G.state_dict(),
                "opt_G": opt_G.state_dict(),
                "F": F.state_dict(),
                "opt_F": opt_F.state_dict(),
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
    if not os.path.isdir("output"):
        os.makedirs("output")
    train(params, config, device)

if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params)