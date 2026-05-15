#!/usr/bin/env python3
"""
GPT training from scratch — PyTorch + Accelerate + wandb.

Single-GPU / CPU:
    python src/train.py --dataset data/shakespeare.txt

Multi-GPU (all visible GPUs):
    accelerate launch src/train.py --dataset data/shakespeare.txt

Mixed precision on CUDA:
    accelerate launch src/train.py --dataset data/shakespeare.txt --fp16

List presets:
    python src/train.py --list_models
"""

import os
import sys
import glob
import json
import math
import argparse
import numpy as np
import torch
from accelerate import Accelerator

import dataset
import tokenizer as tok_module
import configs
from model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--list_models', action='store_true',
                   help='Print all model presets and exit')
    p.add_argument('--dataset',  default=None,
                   help='Text file or directory of .txt/.md files')
    p.add_argument('--model',    default='small-cpu',
                   help='Model preset (see --list_models)')
    p.add_argument('--save_dir', default='checkpoints')

    # Tokenizer
    p.add_argument('--tokenizer', default='char', choices=['char', 'tiktoken'],
                   help='char (trains vocab from data) or tiktoken (fixed GPT-2/4 vocab)')
    p.add_argument('--tiktoken_encoding', default='gpt2',
                   choices=['gpt2', 'cl100k_base', 'o200k_base'],
                   help='tiktoken encoding — gpt2 | cl100k_base | o200k_base')

    # Architecture overrides
    p.add_argument('--n_layer',  type=int,   default=None)
    p.add_argument('--n_head',   type=int,   default=None)
    p.add_argument('--n_embd',   type=int,   default=None)
    p.add_argument('--n_ctx',    type=int,   default=None)
    p.add_argument('--dropout',  type=float, default=0.0)

    # Dataset
    p.add_argument('--stride',        type=int,   default=None,
                   help='Sliding-window stride (default: n_ctx = no overlap)')
    p.add_argument('--val_fraction',  type=float, default=0.1,
                   help='Fraction held out for validation (0 to disable)')

    # Training
    p.add_argument('--batch_size',    type=int,   default=2)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--steps',         type=int,   default=500,
                   help='Max training steps (0 = no limit)')
    p.add_argument('--min_loss',      type=float, default=None,
                   help='Stop when train loss drops below this value')
    p.add_argument('--warmup_steps',  type=int,   default=100)
    p.add_argument('--weight_decay',  type=float, default=0.01)
    p.add_argument('--clip_norm',     type=float, default=1.0,
                   help='Gradient clip norm (0 to disable)')
    p.add_argument('--accum_steps',   type=int,   default=1,
                   help='Gradient accumulation micro-steps per optimizer update')
    p.add_argument('--fp16',          action='store_true',
                   help='Mixed-precision training (requires CUDA)')

    # Logging / saving
    p.add_argument('--save_every',    type=int,   default=100)
    p.add_argument('--eval_every',    type=int,   default=None,
                   help='Evaluate val loss every N steps (default: save_every)')
    p.add_argument('--sample_every',  type=int,   default=50)
    p.add_argument('--sample_length', type=int,   default=200)
    p.add_argument('--seed',          type=int,   default=42)

    # wandb
    p.add_argument('--wandb',         action='store_true',
                   help='Enable Weights & Biases logging')
    p.add_argument('--wandb_project', default='pocketmodel')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_config(args, n_vocab: int) -> GPTConfig:
    preset = configs.get_preset(args.model)
    if args.n_layer is not None: preset['n_layer'] = args.n_layer
    if args.n_head  is not None: preset['n_head']  = args.n_head
    if args.n_embd  is not None: preset['n_embd']  = args.n_embd
    if args.n_ctx   is not None: preset['n_ctx']   = args.n_ctx
    return GPTConfig(n_vocab=n_vocab, dropout=args.dropout, **preset)


def _lr_schedule(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay to min_ratio × peak LR."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if total == 0:
        return 1.0
    progress = min((step - warmup) / max(1, total - warmup), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


@torch.no_grad()
def _sample_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < vals[-1]] = float('-inf')
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_sample(model: GPT, tokenizer, device,
                    length: int = 200, temperature: float = 0.8,
                    top_k: int = 10) -> str:
    model.eval()
    prime = tokenizer.encode('\n') or [0]
    idx = torch.tensor([prime], dtype=torch.long, device=device)

    logits, _, past_kvs = model(idx)
    next_tok = _sample_token(logits[0, -1], temperature, top_k)
    tokens = [next_tok]

    for _ in range(length - 1):
        if past_kvs[0][0].size(2) >= model.config.n_ctx:
            past_kvs = [(k[:, :, 1:], v[:, :, 1:]) for k, v in past_kvs]
        idx = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        logits, _, past_kvs = model(idx, past_kvs=past_kvs)
        next_tok = _sample_token(logits[0, 0], temperature, top_k)
        tokens.append(next_tok)

    model.train()
    return tokenizer.decode(tokens)


@torch.no_grad()
def eval_val_loss(model: GPT, val_chunks: np.ndarray,
                  batch_size: int, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(val_chunks), batch_size):
        batch = torch.tensor(
            val_chunks[i:i + batch_size], dtype=torch.long, device=device
        )
        _, loss, _ = model(batch[:, :-1], targets=batch[:, 1:])
        total += loss.item()
        count += 1
    model.train()
    return total / max(1, count)


def _ckpt_path(save_dir: str, step: int) -> str:
    return os.path.join(save_dir, f'ckpt_{step:07d}.pt')


def _latest_checkpoint(save_dir: str):
    """Return (path, step) of the newest checkpoint, or (None, 0)."""
    ckpts = sorted(glob.glob(os.path.join(save_dir, 'ckpt_*.pt')))
    if not ckpts:
        return None, 0
    path = ckpts[-1]
    step = int(os.path.splitext(os.path.basename(path))[0].split('_')[1])
    return path, step


def _save_checkpoint(accelerator: Accelerator, model, optimizer,
                     scheduler, step: int, save_dir: str):
    raw = accelerator.unwrap_model(model)
    torch.save({
        'step':      step,
        'model':     raw.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, _ckpt_path(save_dir, step))
    print(f'  ✓ Checkpoint saved: {_ckpt_path(save_dir, step)}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.list_models:
        print('\n' + configs.list_presets() + '\n')
        sys.exit(0)

    if args.dataset is None:
        print('[Error] --dataset is required. Use --list_models to see presets.')
        sys.exit(1)

    eval_every = args.eval_every if args.eval_every is not None else args.save_every

    # ── Accelerate setup ─────────────────────────────────────────────────
    accelerator = Accelerator(
        mixed_precision='fp16' if args.fp16 else 'no',
        gradient_accumulation_steps=args.accum_steps,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        os.makedirs(args.save_dir, exist_ok=True)

    # ── wandb (optional) ─────────────────────────────────────────────────
    use_wandb = False
    if args.wandb and accelerator.is_main_process:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
            use_wandb = True
        except ImportError:
            print("[Warning] wandb not installed — pip install wandb, or drop --wandb")

    # ── 1. Data + tokenizer ──────────────────────────────────────────────
    if accelerator.is_main_process:
        print('\n[ Step 1 ] Loading dataset...')
    raw_text = dataset.read_all_text(args.dataset)
    if accelerator.is_main_process:
        print(f'  Characters: {len(raw_text):,}  |  Tokenizer: {args.tokenizer}')

    if args.tokenizer == 'tiktoken':
        tokenizer = tok_module.TiktokenTokenizer(encoding=args.tiktoken_encoding)
        if accelerator.is_main_process:
            print(f'  tiktoken encoding: {args.tiktoken_encoding} '
                  f'({tokenizer.n_vocab:,} tokens)')
    else:
        tokenizer = tok_module.CharTokenizer().build_from_text(raw_text)

    if accelerator.is_main_process:
        tokenizer.save(args.save_dir)

    tokens = tokenizer.encode(raw_text)
    if accelerator.is_main_process:
        print(f'  Total tokens: {len(tokens):,}')

    if args.val_fraction > 0:
        train_tokens, val_tokens = dataset.train_val_split(tokens, args.val_fraction)
    else:
        train_tokens, val_tokens = tokens, []
    if accelerator.is_main_process:
        print(f'  Train: {len(train_tokens):,}  |  Val: {len(val_tokens):,}')

    # ── 2. Model ─────────────────────────────────────────────────────────
    config = build_config(args, tokenizer.n_vocab)
    if accelerator.is_main_process:
        print(f'\n[ Step 2 ] Model: {args.model}')
        approx = configs.estimate_params(
            n_vocab=config.n_vocab, n_ctx=config.n_ctx,
            n_embd=config.n_embd, n_head=config.n_head, n_layer=config.n_layer,
        )
        print(f'  n_vocab={config.n_vocab}  n_ctx={config.n_ctx}  '
              f'n_embd={config.n_embd}  n_head={config.n_head}  n_layer={config.n_layer}')
        print(f'  ~Params: {configs.format_params(approx)}')

    stride = args.stride
    train_chunks = dataset.make_chunks(train_tokens, config.n_ctx, stride=stride)
    val_chunks = (
        dataset.make_chunks(val_tokens, config.n_ctx)
        if len(val_tokens) > config.n_ctx else None
    )
    if accelerator.is_main_process:
        print(f'  Train chunks: {len(train_chunks):,}'
              + (f'  |  Val chunks: {len(val_chunks):,}' if val_chunks is not None else ''))

    if len(train_chunks) < args.batch_size:
        raise ValueError(
            f'Only {len(train_chunks)} chunks but batch_size={args.batch_size}. '
            'Add more text or reduce --n_ctx / --val_fraction.'
        )

    sampler = dataset.Sampler(train_chunks)
    model = GPT(config)
    if accelerator.is_main_process:
        print(f'  Actual params: {model.num_params:,}')

    # ── Optimizer + LR scheduler ─────────────────────────────────────────
    optimizer = model.configure_optimizers(args.learning_rate, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_schedule(step, args.warmup_steps, args.steps),
    )

    # ── Resume from checkpoint ───────────────────────────────────────────
    ckpt_path, start_step = _latest_checkpoint(args.save_dir)
    config_path = os.path.join(args.save_dir, 'config.json')

    if ckpt_path and os.path.exists(config_path):
        with open(config_path) as f:
            saved_cfg = GPTConfig.model_validate(json.load(f))
        if saved_cfg == config:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            if accelerator.is_main_process:
                print(f'\n  Resumed from step {start_step}: {ckpt_path}')
        else:
            start_step = 0
            if accelerator.is_main_process:
                print('\n  [Warning] Checkpoint config mismatch — starting fresh.')
                print('            Use a different --save_dir to keep the old model.')
    elif ckpt_path:
        start_step = 0

    if accelerator.is_main_process:
        with open(config_path, 'w') as f:
            json.dump(config.model_dump(), f, indent=2)

    # ── Accelerate: wrap model, optimizer, scheduler ─────────────────────
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # ── 3. Training loop ─────────────────────────────────────────────────
    step        = start_step
    stop_reason = 'completed'

    if accelerator.is_main_process:
        conds = []
        if args.steps:         conds.append(f'steps={args.steps}')
        if args.min_loss:      conds.append(f'min_loss={args.min_loss}')
        if not conds:          conds.append('∞')
        print(f'\n[ Step 3 ] Training | {" | ".join(conds)} | '
              f'batch={args.batch_size}×{args.accum_steps} | '
              f'lr={args.learning_rate} | warmup={args.warmup_steps} | '
              f'wd={args.weight_decay} | clip={args.clip_norm}')
        print(f'\n{"Step":>6}  {"Loss":>8}  {"Val Loss":>10}  {"LR":>10}')
        print('-' * 42)

    model.train()

    try:
        while True:
            if args.steps and step >= args.steps:
                stop_reason = f'reached --steps {args.steps}'
                break

            # ── Gradient accumulation ────────────────────────────────────
            optimizer.zero_grad()
            micro_losses = []

            for _ in range(args.accum_steps):
                with accelerator.accumulate(model):
                    batch = torch.tensor(
                        sampler.sample(args.batch_size),
                        dtype=torch.long, device=device,
                    )
                    _, loss, _ = model(batch[:, :-1], targets=batch[:, 1:])
                    # Scale loss so accumulated gradient equals the true mean.
                    accelerator.backward(loss / args.accum_steps)
                    micro_losses.append(loss.item())

            if args.clip_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.clip_norm)

            optimizer.step()
            scheduler.step()

            avg_loss   = float(np.mean(micro_losses))
            current_lr = scheduler.get_last_lr()[0]
            step += 1

            if accelerator.is_main_process:
                val_str = ''
                if val_chunks is not None and step % eval_every == 0:
                    vl = eval_val_loss(
                        accelerator.unwrap_model(model),
                        val_chunks, args.batch_size, device,
                    )
                    val_str = f'{vl:>10.4f}'
                    if use_wandb:
                        import wandb
                        wandb.log({'val_loss': vl}, step=step)
                else:
                    val_str = f'{"":>10}'

                print(f'{step:>6}  {avg_loss:>8.4f}  {val_str}  {current_lr:>10.6f}')

                if use_wandb:
                    import wandb
                    wandb.log({'train_loss': avg_loss, 'lr': current_lr}, step=step)

            if args.min_loss is not None and avg_loss <= args.min_loss:
                stop_reason = f'loss {avg_loss:.4f} ≤ --min_loss {args.min_loss}'
                break

            if accelerator.is_main_process and step % args.sample_every == 0:
                raw_model = accelerator.unwrap_model(model)
                sample = generate_sample(raw_model, tokenizer, device,
                                         args.sample_length)
                print(f'\n{"─"*50}  sample @ step {step}')
                print(sample)
                print('─' * 50 + '\n')

            if accelerator.is_main_process and step % args.save_every == 0:
                _save_checkpoint(accelerator, model, optimizer, scheduler,
                                 step, args.save_dir)

    except KeyboardInterrupt:
        stop_reason = 'interrupted by user'
        if accelerator.is_main_process:
            print('\n\n  Interrupted — saving checkpoint...')

    finally:
        if step > start_step and accelerator.is_main_process:
            _save_checkpoint(accelerator, model, optimizer, scheduler,
                             step, args.save_dir)
            print(f'\n[ Done ] {stop_reason}. Step {step}.')
            if use_wandb:
                import wandb
                wandb.finish()


if __name__ == '__main__':
    main()
