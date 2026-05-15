#!/usr/bin/env python3
"""
GPT-2/GPT-3 training from scratch — no pre-trained weights or external vocab files needed.

Usage:
    # Default (char tokenizer, small-cpu preset)
    python src/train.py --dataset data/shakespeare.txt

    # BPE tokenizer with 800 subword pieces
    python src/train.py --dataset data/shakespeare.txt --tokenizer bpe --vocab_size 800

    # GPT-2 small with sliding-window chunks, validation, gradient accumulation
    python src/train.py --dataset data/shakespeare.txt --model gpt2-small \
        --stride 512 --accum_steps 4 --val_fraction 0.1

    # List all available presets
    python src/train.py --list_models
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import tensorflow as tf

import model
import dataset
import tokenizer as tok_module
import configs

tf1 = tf.compat.v1
tf1.disable_eager_execution()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--list_models',   action='store_true', help='Print all model presets and exit')
    p.add_argument('--dataset',       default=None,        help='Text file or directory of .txt files')
    p.add_argument('--model',         default='small-cpu', help='Model preset name (see --list_models)')
    p.add_argument('--save_dir',      default='checkpoints')

    # Tokenizer
    p.add_argument('--tokenizer',     default='char',      choices=['char', 'bpe'],
                   help='Tokenizer backend: char (default) or bpe (subword)')
    p.add_argument('--vocab_size',    type=int, default=1000,
                   help='BPE vocabulary size (ignored for --tokenizer char)')

    # Architecture overrides — take priority over the preset
    p.add_argument('--n_layer',  type=int, default=None)
    p.add_argument('--n_head',   type=int, default=None)
    p.add_argument('--n_embd',   type=int, default=None)
    p.add_argument('--n_ctx',    type=int, default=None)

    # Dataset
    p.add_argument('--stride',        type=int,   default=None,
                   help='Sliding-window stride for chunks (default: n_ctx = no overlap)')
    p.add_argument('--val_fraction',  type=float, default=0.1,
                   help='Fraction of tokens held out for validation (0 to disable)')

    # Training knobs
    p.add_argument('--batch_size',    type=int,   default=2)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--steps',         type=int,   default=500,
                   help='Max training steps (0 = no limit)')
    p.add_argument('--min_loss',      type=float, default=None,
                   help='Stop early when train loss drops below this value')
    p.add_argument('--warmup_steps',  type=int,   default=100,
                   help='Linear LR warmup steps before cosine decay')
    p.add_argument('--weight_decay',  type=float, default=0.01,
                   help='AdamW weight decay (0 to disable)')
    p.add_argument('--clip_norm',     type=float, default=1.0,
                   help='Global gradient clip norm (0 to disable)')
    p.add_argument('--accum_steps',   type=int,   default=1,
                   help='Gradient accumulation micro-steps per optimizer update')
    p.add_argument('--fp16',          action='store_true',
                   help='Enable mixed-precision training (requires GPU + TF >= 2.4)')

    # Logging / saving
    p.add_argument('--save_every',    type=int,   default=100)
    p.add_argument('--eval_every',    type=int,   default=None,
                   help='Evaluate val loss every N steps (default: same as save_every)')
    p.add_argument('--sample_every',  type=int,   default=50,
                   help='Print a generated sample every N steps')
    p.add_argument('--sample_length', type=int,   default=200,
                   help='Tokens to generate per sample')
    p.add_argument('--seed',          type=int,   default=42)
    return p.parse_args()


def resolve_hparams(args, n_vocab):
    preset = configs.get_preset(args.model)
    if args.n_layer is not None: preset['n_layer'] = args.n_layer
    if args.n_head  is not None: preset['n_head']  = args.n_head
    if args.n_embd  is not None: preset['n_embd']  = args.n_embd
    if args.n_ctx   is not None: preset['n_ctx']   = args.n_ctx
    if preset['n_embd'] % preset['n_head'] != 0:
        raise ValueError(
            f"n_embd ({preset['n_embd']}) must be divisible by n_head ({preset['n_head']})"
        )
    hp = model.default_hparams()
    hp.override_from_dict({**preset, 'n_vocab': n_vocab})
    return hp


def compute_lr(step, base_lr, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Linear warmup then cosine decay to min_lr_ratio * base_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if total_steps == 0:
        # Unlimited training: stay at base_lr after warmup
        return base_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _sample_token(logits, temperature, top_k):
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


def generate_sample(sess, hparams, tokenizer, length,
                    X_infer_p, infer_p_logits, infer_p_present,
                    X_infer_n, infer_n_logits, infer_n_present, infer_past_ph,
                    temperature=0.8, top_k=10):
    """Generate `length` tokens from a newline seed using the KV cache."""
    prime = tokenizer.encode('\n') or [0]
    prompt_arr = np.array([prime], dtype=np.int32)

    # Process the seed prompt in one forward pass
    logits_val, past_np = sess.run(
        [infer_p_logits, infer_p_present],
        feed_dict={X_infer_p: prompt_arr},
    )

    next_tok = _sample_token(logits_val[0, -1, :], temperature, top_k)
    generated = [next_tok]

    for _ in range(length - 1):
        # Trim KV cache when it reaches the context limit
        if past_np.shape[4] >= hparams.n_ctx:
            past_np = past_np[:, :, :, :, -(hparams.n_ctx - 1):, :]

        tok_arr = np.array([[next_tok]], dtype=np.int32)
        logits_val, new_present = sess.run(
            [infer_n_logits, infer_n_present],
            feed_dict={X_infer_n: tok_arr, infer_past_ph: past_np},
        )
        past_np = np.concatenate([past_np, new_present], axis=4)
        next_tok = _sample_token(logits_val[0, 0, :], temperature, top_k)
        generated.append(next_tok)

    return tokenizer.decode(generated)


def _hparams_match(save_dir, hparams):
    path = os.path.join(save_dir, 'hparams.json')
    if not os.path.exists(path):
        return False
    with open(path) as f:
        saved = json.load(f)
    keys = ('n_vocab', 'n_ctx', 'n_embd', 'n_head', 'n_layer')
    return all(saved.get(k) == getattr(hparams, k) for k in keys)


def _eval_val_loss(sess, val_loss_op, X_val, val_chunks, batch_size):
    total, count = 0.0, 0
    for i in range(0, len(val_chunks), batch_size):
        batch = val_chunks[i : i + batch_size]
        if len(batch) == 0:
            continue
        total += sess.run(val_loss_op, feed_dict={X_val: batch})
        count += 1
    return total / max(1, count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.list_models:
        print("\nAvailable model presets:\n")
        print(configs.list_presets())
        print()
        sys.exit(0)

    if args.dataset is None:
        print("[Error] --dataset is required. Use --list_models to see presets.")
        sys.exit(1)

    eval_every = args.eval_every if args.eval_every is not None else args.save_every

    np.random.seed(args.seed)
    tf1.set_random_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 1. Load and tokenize data ────────────────────────────────────────
    print("\n[ Step 1 ] Loading dataset...")
    raw_text = dataset.read_all_text(args.dataset)
    print(f"  Total characters: {len(raw_text):,}")

    print(f"\n  Tokenizer: {args.tokenizer}")
    if args.tokenizer == 'bpe':
        tokenizer = tok_module.BPETokenizer().build_from_text(raw_text, vocab_size=args.vocab_size)
    else:
        tokenizer = tok_module.CharTokenizer().build_from_text(raw_text)
    tokenizer.save(args.save_dir)

    tokens = tokenizer.encode(raw_text)
    print(f"  Total tokens: {len(tokens):,}")

    # Train / validation split
    val_chunks = None
    if args.val_fraction > 0:
        train_tokens, val_tokens = dataset.train_val_split(tokens, args.val_fraction)
        print(f"  Train tokens: {len(train_tokens):,}  |  Val tokens: {len(val_tokens):,}")
    else:
        train_tokens, val_tokens = tokens, []
        print(f"  (Validation disabled — val_fraction=0)")

    # ── 2. Resolve hyperparameters ───────────────────────────────────────
    print(f"\n[ Step 2 ] Building model  (preset: {args.model})")
    hparams = resolve_hparams(args, tokenizer.n_vocab)

    approx = configs.estimate_params(
        n_vocab=hparams.n_vocab, n_ctx=hparams.n_ctx,
        n_embd=hparams.n_embd, n_head=hparams.n_head, n_layer=hparams.n_layer,
    )
    print(f"  n_vocab={hparams.n_vocab}  n_ctx={hparams.n_ctx}  "
          f"n_embd={hparams.n_embd}  n_head={hparams.n_head}  n_layer={hparams.n_layer}")
    print(f"  Estimated parameters: ~{configs.format_params(approx)}")

    stride = args.stride  # None → defaults to n_ctx inside make_chunks
    train_chunks = dataset.make_chunks(train_tokens, hparams.n_ctx, stride=stride)
    print(f"  Train chunks (ctx={hparams.n_ctx}, stride={stride or hparams.n_ctx}): {len(train_chunks):,}")

    if len(val_tokens) > hparams.n_ctx:
        val_chunks = dataset.make_chunks(val_tokens, hparams.n_ctx, stride=stride)
        print(f"  Val   chunks: {len(val_chunks):,}")

    if len(train_chunks) < args.batch_size:
        raise ValueError(
            f"Not enough data: only {len(train_chunks)} chunks but batch_size={args.batch_size}. "
            f"Add more text or reduce --n_ctx / --val_fraction."
        )

    sampler = dataset.Sampler(train_chunks)
    can_resume = _hparams_match(args.save_dir, hparams)

    with open(os.path.join(args.save_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams.__dict__, f, indent=2)

    with tf1.Session(graph=tf1.Graph()) as sess:

        # ── Training graph ──────────────────────────────────────────────
        X_train  = tf1.placeholder(tf.int32, [None, hparams.n_ctx + 1], name='X_train')
        lr_ph    = tf1.placeholder(tf.float32, shape=[], name='lr')

        lm_out       = model.model(hparams=hparams, X=X_train[:, :-1])
        train_logits = lm_out['logits']

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=X_train[:, 1:], logits=train_logits,
            )
        )

        # Optimizer (AdamW = Adam + decoupled weight decay)
        optimizer = tf1.train.AdamOptimizer(learning_rate=lr_ph)
        if args.fp16:
            try:
                optimizer = tf1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
                print("  Mixed precision (fp16) enabled")
            except Exception as e:
                print(f"  [Warning] fp16 not available in this TF build: {e}")

        all_gv   = optimizer.compute_gradients(loss)
        gv_pairs = [(g, v) for g, v in all_gv if g is not None]
        grads    = [g for g, _ in gv_pairs]
        tvars    = [v for _, v in gv_pairs]

        # Gradient accumulation: accumulate micro-batch gradients before applying
        accum_vars = [
            tf.Variable(tf.zeros_like(v), trainable=False, name=f'accum_{i}')
            for i, v in enumerate(tvars)
        ]
        zero_op  = tf.group(*[av.assign(tf.zeros_like(av)) for av in accum_vars])
        accum_op = tf.group(*[av.assign_add(g) for av, g in zip(accum_vars, grads)])

        # Clip the *averaged* accumulated gradient, then apply
        averaged = [av / float(args.accum_steps) for av in accum_vars]
        if args.clip_norm > 0:
            clipped, _ = tf.clip_by_global_norm(averaged, args.clip_norm)
        else:
            clipped = averaged

        apply_op = optimizer.apply_gradients(zip(clipped, tvars))

        # Decoupled weight decay (skip bias 'b:*' and LayerNorm scale 'g:*')
        decay_vars = [
            v for v in tvars
            if not v.name.split('/')[-1].startswith(('b:', 'g:'))
        ]
        if decay_vars and args.weight_decay > 0:
            with tf.control_dependencies([apply_op]):
                update_op = tf.group(*[
                    v.assign(v * (1.0 - lr_ph * args.weight_decay))
                    for v in decay_vars
                ])
        else:
            update_op = apply_op

        # ── Validation graph (shares weights via AUTO_REUSE) ────────────
        X_val    = tf1.placeholder(tf.int32, [None, hparams.n_ctx + 1], name='X_val')
        val_out  = model.model(hparams=hparams, X=X_val[:, :-1])
        val_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=X_val[:, 1:], logits=val_out['logits'],
            )
        )

        # ── KV-cache inference graph (for training samples) ─────────────
        # Prompt pass: process arbitrary-length prompt, return KV cache
        X_infer_p      = tf1.placeholder(tf.int32, [1, None], name='X_infer_p')
        out_infer_p    = model.model(hparams=hparams, X=X_infer_p)
        infer_p_logits  = out_infer_p['logits']
        infer_p_present = out_infer_p['present']

        # Incremental pass: one new token + existing KV cache
        X_infer_n      = tf1.placeholder(tf.int32, [1, 1], name='X_infer_n')
        infer_past_ph  = tf1.placeholder(
            tf.float32,
            [1, hparams.n_layer, 2, hparams.n_head, None, hparams.n_embd // hparams.n_head],
            name='infer_past',
        )
        out_infer_n    = model.model(hparams=hparams, X=X_infer_n, past=infer_past_ph)
        infer_n_logits  = out_infer_n['logits']
        infer_n_present = out_infer_n['present']

        saver = tf1.train.Saver(max_to_keep=3)
        sess.run(tf1.global_variables_initializer())

        ckpt = tf1.train.latest_checkpoint(args.save_dir)
        if ckpt and can_resume:
            print(f"\n  Resuming from checkpoint: {ckpt}")
            saver.restore(sess, ckpt)
        elif ckpt and not can_resume:
            print(f"\n  [Warning] Checkpoint found but hparams differ — starting fresh.")
            print(f"            Use a different --save_dir to keep the old model.\n")

        actual_params = sum(
            np.prod(v.get_shape().as_list())
            for v in tf1.trainable_variables()
            if not v.name.startswith('accum_')
        )
        print(f"  Actual trainable parameters: {actual_params:,}")

        # ── 3. Training loop ─────────────────────────────────────────────
        stop_conds = []
        if args.steps:            stop_conds.append(f"steps={args.steps}")
        if args.min_loss is not None: stop_conds.append(f"min_loss={args.min_loss}")
        if not stop_conds:        stop_conds.append("∞")

        print(f"\n[ Step 3 ] Training | {' | '.join(stop_conds)} | "
              f"batch={args.batch_size}×{args.accum_steps} | "
              f"lr={args.learning_rate} | warmup={args.warmup_steps} | "
              f"wd={args.weight_decay} | clip={args.clip_norm}\n")
        print(f"{'Step':>6}  {'Loss':>8}  {'LR':>10}")
        print("-" * 30)

        step        = 0
        stop_reason = "completed"

        try:
            while True:
                if args.steps and step >= args.steps:
                    stop_reason = f"reached --steps {args.steps}"
                    break

                # --- Accumulate gradients over accum_steps micro-batches ---
                micro_losses = []
                for _ in range(args.accum_steps):
                    batch    = sampler.sample(args.batch_size)
                    loss_val, _ = sess.run([loss, accum_op], feed_dict={X_train: batch})
                    micro_losses.append(loss_val)

                lr_val = compute_lr(step, args.learning_rate, args.warmup_steps, args.steps)
                sess.run(update_op, feed_dict={lr_ph: lr_val})
                sess.run(zero_op)

                avg_loss = float(np.mean(micro_losses))
                step += 1
                print(f"{step:>6}  {avg_loss:>8.4f}  {lr_val:>10.6f}")

                if args.min_loss is not None and avg_loss <= args.min_loss:
                    stop_reason = f"loss {avg_loss:.4f} reached --min_loss {args.min_loss}"
                    break

                if step % args.sample_every == 0:
                    sample_text = generate_sample(
                        sess, hparams, tokenizer, args.sample_length,
                        X_infer_p, infer_p_logits, infer_p_present,
                        X_infer_n, infer_n_logits, infer_n_present, infer_past_ph,
                    )
                    print("\n" + "─" * 50 + f"  sample @ step {step}")
                    print(sample_text)
                    print("─" * 50 + "\n")

                if val_chunks is not None and step % eval_every == 0:
                    vl = _eval_val_loss(sess, val_loss, X_val, val_chunks, args.batch_size)
                    print(f"  ✦ Val loss: {vl:.4f}\n")

                if step % args.save_every == 0:
                    path = saver.save(
                        sess,
                        os.path.join(args.save_dir, 'model'),
                        global_step=step,
                    )
                    print(f"  ✓ Checkpoint saved: {path}\n")

        except KeyboardInterrupt:
            stop_reason = "interrupted by user"
            print("\n\n  Interrupted — saving final checkpoint...")

        if step > 0:
            path = saver.save(
                sess,
                os.path.join(args.save_dir, 'model'),
                global_step=step,
            )
            print(f"\n[ Done ] Stopped: {stop_reason}. Step {step}. Final checkpoint: {path}")
        else:
            print("\n[ Done ] No steps run — no checkpoint saved.")


if __name__ == '__main__':
    main()
