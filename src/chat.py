#!/usr/bin/env python3
"""
Interactive chat with your trained GPT model — PyTorch edition.

Tokens are streamed to the terminal as they are generated.
KV cache is maintained across decoding steps for O(1)-per-token generation.

Usage:
    python src/chat.py
    python src/chat.py --checkpoints_dir checkpoints --length 300 --temperature 0.8
"""

import os
import glob
import json
import argparse
import torch

import tokenizer as tok_module
from model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints_dir', default='checkpoints',
                   help='Directory with saved model + tokenizer')
    p.add_argument('--length',      type=int,   default=200,
                   help='Max tokens to generate per response')
    p.add_argument('--temperature', type=float, default=0.8,
                   help='Higher = more creative, lower = more focused')
    p.add_argument('--top_k',       type=int,   default=10,
                   help='Sample from top-k most likely tokens (0 = no truncation)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoints_dir: str, device: torch.device):
    config_path = os.path.join(checkpoints_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in '{checkpoints_dir}'.")
    with open(config_path) as f:
        config = GPTConfig.model_validate(json.load(f))

    ckpts = sorted(glob.glob(os.path.join(checkpoints_dir, 'ckpt_*.pt')))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint (ckpt_*.pt) found in '{checkpoints_dir}'.")

    latest = ckpts[-1]
    gpt = GPT(config)
    ckpt = torch.load(latest, map_location='cpu', weights_only=True)
    gpt.load_state_dict(ckpt['model'])
    gpt.to(device).eval()
    return gpt, config, latest


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sample_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < vals[-1]] = float('-inf')
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_streaming(model: GPT, tokenizer, device,
                       prompt_tokens: list[int], length: int,
                       temperature: float, top_k: int):
    """
    Yield decoded strings one token at a time using a KV cache.

    The full prompt is processed in one forward pass to build the initial
    cache.  Each subsequent token then does a single O(1) incremental step
    that extends the cache by one position instead of recomputing the whole
    sequence.
    """
    tokens = prompt_tokens or [0]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    # Full prompt forward pass — populates the KV cache for all prompt positions.
    logits, _, past_kvs = model(idx)
    next_tok = _sample_token(logits[0, -1], temperature, top_k)
    yield tokenizer.decode([next_tok])

    for _ in range(length - 1):
        # Trim the oldest cached position when we reach the context limit.
        if past_kvs[0][0].size(2) >= model.config.n_ctx:
            past_kvs = [(k[:, :, 1:], v[:, :, 1:]) for k, v in past_kvs]

        idx = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        logits, _, past_kvs = model(idx, past_kvs=past_kvs)
        next_tok = _sample_token(logits[0, 0], temperature, top_k)
        yield tokenizer.decode([next_tok])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load tokenizer ───────────────────────────────────────────────────
    try:
        tokenizer = tok_module.load_tokenizer(args.checkpoints_dir)
    except FileNotFoundError as e:
        print(f'[Error] {e}')
        print('  Train the model first:  python src/train.py --dataset data/sample.txt')
        return
    print(f'Tokenizer loaded  ({tokenizer.n_vocab:,} vocab)')

    # ── Load model ───────────────────────────────────────────────────────
    try:
        model, config, ckpt_path = load_model(args.checkpoints_dir, device)
    except FileNotFoundError as e:
        print(f'[Error] {e}')
        print('  Train the model first:  python src/train.py --dataset data/sample.txt')
        return
    print(f'Model loaded from {ckpt_path}  ({model.num_params:,} params, device={device})')
    print(f'Settings → temperature={args.temperature}  top_k={args.top_k}  '
          f'length={args.length}')
    print('\nType a prompt and press Enter. The model streams its continuation.')
    print('Commands:  /temp <value>   /topk <value>   /length <value>   /quit\n')
    print('─' * 60)

    temperature = args.temperature
    top_k       = args.top_k
    length      = args.length

    while True:
        try:
            prompt = input('\nYou › ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye!')
            break

        if not prompt:
            continue

        # ── Runtime commands ─────────────────────────────────────────────
        if prompt.startswith('/quit'):
            print('Goodbye!')
            break
        elif prompt.startswith('/temp '):
            try:
                temperature = float(prompt.split()[1])
                print(f'  temperature → {temperature}')
            except ValueError:
                print('  Usage: /temp 0.8')
            continue
        elif prompt.startswith('/topk '):
            try:
                top_k = int(prompt.split()[1])
                print(f'  top_k → {top_k}')
            except ValueError:
                print('  Usage: /topk 10')
            continue
        elif prompt.startswith('/length '):
            try:
                length = int(prompt.split()[1])
                print(f'  length → {length}')
            except ValueError:
                print('  Usage: /length 200')
            continue

        # ── Generate (streaming) ─────────────────────────────────────────
        prompt_tokens = tokenizer.encode(prompt)
        if not prompt_tokens:
            print('  [Warning] Prompt contained no known tokens.')
            continue

        print('\nModel › ', end='', flush=True)
        for piece in generate_streaming(
            model, tokenizer, device,
            prompt_tokens, length, temperature, top_k,
        ):
            print(piece, end='', flush=True)
        print('\n' + '─' * 60)


if __name__ == '__main__':
    main()
