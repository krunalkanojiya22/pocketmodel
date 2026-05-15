#!/usr/bin/env python3
"""
Interactive chat with your trained GPT model.

Tokens are streamed to the terminal as they are generated.
KV-cache (past key/value tensors) is reused across tokens so each
generation step is O(1) in sequence length instead of O(n²).

Usage:
    python src/chat.py
    python src/chat.py --checkpoints_dir checkpoints --length 300 --temperature 0.8
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf

import model
import tokenizer as tok_module

tf1 = tf.compat.v1
tf1.disable_eager_execution()


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


def load_hparams(checkpoints_dir):
    path = os.path.join(checkpoints_dir, 'hparams.json')
    hp = model.default_hparams()
    if os.path.exists(path):
        with open(path) as f:
            hp.override_from_dict(json.load(f))
    return hp


def _sample_token(logits, temperature, top_k):
    """Sample one token from logits using temperature + top-k."""
    if top_k > 0:
        top_indices = np.argsort(logits)[-top_k:]
        mask = np.full_like(logits, -1e10)
        mask[top_indices] = logits[top_indices]
        logits = mask
    logits = logits / max(temperature, 1e-8)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate_streaming(sess, tokenizer, hparams,
                       X_prompt, prompt_logits, prompt_present,
                       X_next, next_logits, next_present, past_ph,
                       prompt_tokens, length, temperature, top_k):
    """
    Yield decoded strings one token at a time using a KV cache.

    On the first call the full prompt is processed in a single forward
    pass and the resulting key/value tensors are stored in `past_np`.
    Subsequent tokens each take a single O(1) incremental step that
    reads one new token and the accumulated cache rather than
    re-processing the whole sequence.
    """
    tokens = list(prompt_tokens) or [0]
    prompt_arr = np.array([tokens], dtype=np.int32)

    # Forward pass over the entire prompt — produces KV cache for all positions
    logits_val, past_np = sess.run(
        [prompt_logits, prompt_present],
        feed_dict={X_prompt: prompt_arr},
    )

    # Sample first generated token from the last prompt position
    next_tok = _sample_token(logits_val[0, -1, :], temperature, top_k)
    yield tokenizer.decode([next_tok])

    for _ in range(length - 1):
        # Slide the window when we reach the context limit
        if past_np.shape[4] >= hparams.n_ctx:
            past_np = past_np[:, :, :, :, -(hparams.n_ctx - 1):, :]

        tok_arr = np.array([[next_tok]], dtype=np.int32)
        logits_val, new_present = sess.run(
            [next_logits, next_present],
            feed_dict={X_next: tok_arr, past_ph: past_np},
        )
        # Extend the KV cache along the sequence axis
        past_np = np.concatenate([past_np, new_present], axis=4)

        next_tok = _sample_token(logits_val[0, 0, :], temperature, top_k)
        yield tokenizer.decode([next_tok])


def main():
    args = parse_args()

    # ── Load tokenizer ───────────────────────────────────────────────────
    tok_path = os.path.join(args.checkpoints_dir, 'tokenizer.json')
    if not os.path.exists(tok_path):
        print(f"[Error] tokenizer.json not found in '{args.checkpoints_dir}'.")
        print("  Train the model first:  python src/train.py --dataset data/sample.txt")
        return

    tokenizer = tok_module.load_tokenizer(args.checkpoints_dir)
    print(f"Tokenizer loaded  ({tokenizer.n_vocab} tokens in vocab)")

    # ── Load hparams ─────────────────────────────────────────────────────
    hparams = load_hparams(args.checkpoints_dir)

    # ── Build inference graph ────────────────────────────────────────────
    with tf1.Session(graph=tf1.Graph()) as sess:

        # Prompt pass: process arbitrary-length input, return logits + KV cache
        X_prompt       = tf1.placeholder(tf.int32, [1, None], name='X_prompt')
        out_prompt     = model.model(hparams=hparams, X=X_prompt)
        prompt_logits  = out_prompt['logits']
        prompt_present = out_prompt['present']

        # Incremental pass: one new token + accumulated KV cache
        X_next        = tf1.placeholder(tf.int32, [1, 1], name='X_next')
        past_ph       = tf1.placeholder(
            tf.float32,
            [1, hparams.n_layer, 2, hparams.n_head, None, hparams.n_embd // hparams.n_head],
            name='past',
        )
        out_next      = model.model(hparams=hparams, X=X_next, past=past_ph)
        next_logits   = out_next['logits']
        next_present  = out_next['present']

        saver = tf1.train.Saver(var_list=tf1.trainable_variables())
        ckpt  = tf1.train.latest_checkpoint(args.checkpoints_dir)
        if not ckpt:
            print(f"[Error] No checkpoint found in '{args.checkpoints_dir}'.")
            print("  Train the model first:  python src/train.py --dataset data/sample.txt")
            return

        saver.restore(sess, ckpt)
        print(f"Model loaded from {ckpt}")
        print(f"Settings  → temperature={args.temperature}  top_k={args.top_k}  length={args.length}")
        print("\nType a prompt and press Enter. The model will stream its continuation.")
        print("Commands:  /temp <value>   /topk <value>   /length <value>   /quit\n")
        print("─" * 60)

        temperature = args.temperature
        top_k       = args.top_k
        length      = args.length

        while True:
            try:
                prompt = input("\nYou › ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not prompt:
                continue

            # ── Commands ─────────────────────────────────────────────────
            if prompt.startswith('/quit'):
                print("Goodbye!")
                break
            elif prompt.startswith('/temp '):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"  temperature → {temperature}")
                except ValueError:
                    print("  Usage: /temp 0.8")
                continue
            elif prompt.startswith('/topk '):
                try:
                    top_k = int(prompt.split()[1])
                    print(f"  top_k → {top_k}")
                except ValueError:
                    print("  Usage: /topk 10")
                continue
            elif prompt.startswith('/length '):
                try:
                    length = int(prompt.split()[1])
                    print(f"  length → {length}")
                except ValueError:
                    print("  Usage: /length 200")
                continue

            # ── Generate (streaming) ─────────────────────────────────────
            prompt_tokens = tokenizer.encode(prompt)
            if not prompt_tokens:
                print("  [Warning] Prompt contained no known tokens.")
                continue

            print("\nModel › ", end='', flush=True)
            for piece in generate_streaming(
                sess, tokenizer, hparams,
                X_prompt, prompt_logits, prompt_present,
                X_next, next_logits, next_present, past_ph,
                prompt_tokens, length, temperature, top_k,
            ):
                print(piece, end='', flush=True)
            print("\n" + "─" * 60)


if __name__ == '__main__':
    main()
