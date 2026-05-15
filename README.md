# pocketmodel

Train a GPT-style language model entirely from your own text — no pre-trained weights, no cloud, no external vocab files. Scales from a tiny CPU model up to LLaMA-3 / Mistral architectures.

Built on **PyTorch 2.x**, **Accelerate**, **tiktoken**, and **Pydantic**.

---

## Architecture

Modern frontier transformer (same family as LLaMA-3, Mistral, Claude):

| Component | Implementation | Why |
|---|---|---|
| Position encoding | **RoPE** — no learned table | Generalises beyond training length; used by every modern LLM |
| Attention | **GQA** — fewer KV heads than Q heads | Shrinks KV cache 4–8×; critical for long context and large batches |
| MLP | **SwiGLU** — gated linear unit | Better optimisation landscape than GELU; used by LLaMA, PaLM, Claude |
| Normalisation | **RMSNorm** — variance only | 20–30% faster than LayerNorm, equally stable |
| Linear biases | **None** | Modern standard; no quality loss |
| Inference | **KV-cache** — O(1) per token | Tokens stream without recomputing past positions |
| Memory | **Gradient checkpointing** | Trade 30% compute for ~10× memory — enables large-batch training |

---

## Features

- **Two tokenizer backends** — character-level (trains from your data) or tiktoken byte-level BPE (GPT-2/4 vocab, 10× faster)
- **Flash Attention** — `F.scaled_dot_product_attention` dispatches to FA2 kernels on CUDA automatically
- **Sliding-window chunks** — overlapping context windows for more training signal per token
- **Train / validation split** — held-out eval loss tracked throughout training
- **Linear warmup + cosine LR decay** — standard schedule used by GPT-2/3/LLaMA
- **AdamW** — decoupled weight decay; biases and RMSNorm params excluded automatically
- **Gradient clipping** — global norm clip prevents explosive updates
- **Gradient accumulation** — simulate large batches without extra memory
- **BF16 / FP16 mixed precision** — BF16 recommended for Ampere+ GPUs
- **Multi-GPU training** — zero code changes; just run with `accelerate launch`
- **torch.compile** — ~2× throughput on Linux/CUDA with one flag
- **Gradient checkpointing** — ~10× memory saving at 30% compute cost
- **Resume training** — automatic checkpoint detection with config compatibility check
- **Weights & Biases** — optional experiment tracking with `--wandb`
- **Pre-tokenised binary** — skip re-tokenisation on large datasets with `--dataset_bin`
- **Pydantic config** — typed, validated model config with JSON serialisation

---

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager

---

## Setup

```bash
git clone <your-repo-url>
cd pocketmodel

# Creates .venv and installs all dependencies
uv sync
```

---

## Quick Start

### 1. Get a dataset

```bash
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
     -o data/shakespeare.txt
```

### 2. Train

```bash
uv run python src/train.py --dataset data/shakespeare.txt
```

### 3. Chat

```bash
uv run python src/chat.py
```

```
You › To be, or not to be

Model › that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows...
```

> **Note:** If you have checkpoints from a previous version of pocketmodel (pre-v0.2), delete them first — the weight names changed with the architecture upgrade.
> ```bash
> rm -rf checkpoints/
> ```

---

## Training

### Basic usage

```bash
uv run python src/train.py --dataset data/shakespeare.txt
```

---

### Tokenizer

| Flag | Value | Description |
|---|---|---|
| `--tokenizer` | `char` *(default)* | Character-level; vocabulary built from training data |
| `--tokenizer` | `tiktoken` | Byte-level BPE; fixed GPT-2/4 vocab, no training step needed |
| `--tiktoken_encoding` | `gpt2` *(default)* | 50,257 tokens — matches GPT-2/3 |
| `--tiktoken_encoding` | `cl100k_base` | 100,277 tokens — GPT-4, better for code and multilingual text |
| `--tiktoken_encoding` | `o200k_base` | 200,019 tokens — GPT-4o, widest coverage |

```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --tokenizer tiktoken --tiktoken_encoding cl100k_base
```

**When to use which:**

| Dataset size | Recommendation |
|---|---|
| < 500 KB | `char` — simple, trains vocab from data |
| 500 KB – 10 MB | Either; tiktoken trains faster |
| > 10 MB | `tiktoken` — far more efficient, richer token semantics |

---

### Model presets

```bash
uv run python src/train.py --list_models
```

```
Preset           n_layer  n_head  n_kv  n_embd   n_ctx   ~Params (vocab=50k)
------------------------------------------------------------------------------
tiny                   2       2    2       64     128                 3.3 M
small-cpu              4       4    4      128     128                 7.3 M   ← default
gpt2-small            12      12   12      768    1024               123.4 M
gpt2-medium           24      16   16     1024    1024               359.5 M
gpt2-large            36      20   20     1280    1024               795.5 M
gpt2-xl               48      25   25     1600    1024                1.57 B
gpt3-small            12      12   12      768    2048               123.4 M
gpt3-medium           24      16   16     1024    2048               359.5 M
gpt3-xl               24      16   16     2048    2048                1.34 B
gpt3-6.7b             32      32   32     4096    2048                6.68 B
gpt3-13b              40      40   40     5120    2048               12.94 B
llama3-1b             16      32    8*    2048    8192               823.9 M
llama3-3b             28      24    8*    3072    8192                2.97 B
llama3-8b             32      32    8*    4096    8192                5.88 B
llama3-70b            80      64    8*    8192    8192               55.78 B
mistral-7b            32      32    8*    4096   32768                5.88 B

* = GQA (n_kv_head < n_head) — smaller KV cache
```

> **CPU training:** use `tiny` or `small-cpu`. Anything above `gpt2-small` needs a GPU.

---

### Stopping conditions

`--steps` and `--min_loss` can be combined — training stops when **either** triggers:

```bash
# Stop after 2000 steps
uv run python src/train.py --dataset data/shakespeare.txt --steps 2000

# Stop when loss drops below 1.2 (no step limit)
uv run python src/train.py --dataset data/shakespeare.txt --steps 0 --min_loss 1.2

# Stop at 5000 steps OR loss 1.2 — whichever comes first
uv run python src/train.py --dataset data/shakespeare.txt --steps 5000 --min_loss 1.2
```

Ctrl+C saves a checkpoint before exiting.

#### Target loss values

| Loss | What it means |
|---|---|
| ~4.2 | Random guessing (65-char vocab) |
| ~2.5 | Basic letter/word frequency learned |
| ~1.5 | Recognising common words and patterns |
| ~1.2 | Readable text with consistent style |
| < 0.5 | Overfitting — model is memorising the data |

---

### Precision

BF16 is recommended over FP16 for large models — same memory, but the wider exponent range prevents overflow at scale.

```bash
# BF16 — Ampere+ GPUs (A100, RTX 30xx/40xx), recommended
accelerate launch src/train.py --dataset data/shakespeare.txt --bf16

# FP16 — older CUDA GPUs (V100, RTX 20xx)
accelerate launch src/train.py --dataset data/shakespeare.txt --fp16
```

---

### torch.compile

Compiles the forward pass into an optimised CUDA graph — roughly **2× throughput** at no memory cost.

```bash
uv run python src/train.py --dataset data/shakespeare.txt --compile
```

> Requires Linux + PyTorch 2.x. Windows support is limited. Adds ~60s warmup on the first step.

---

### Gradient checkpointing

Recomputes activations during backward instead of storing them. Enables **~10× larger effective batch sizes** at a 30–40% compute overhead — essential when training models above `gpt2-small` on a single GPU.

```bash
uv run python src/train.py --dataset data/shakespeare.txt --grad_checkpoint
```

---

### Multi-GPU training

No code changes required:

```bash
# All visible GPUs
accelerate launch src/train.py --dataset data/shakespeare.txt --bf16

# Configure GPUs, nodes, precision interactively
accelerate config
accelerate launch src/train.py --dataset data/shakespeare.txt
```

---

### Learning rate schedule

Linear warmup + cosine decay is enabled by default. LR rises from 0 to `--learning_rate` over `--warmup_steps` steps, then decays to 10% of peak.

```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --learning_rate 3e-4 --warmup_steps 500
```

> For LLaMA-scale models, `3e-4` with longer warmup (`500–2000` steps) is typical.

---

### AdamW

Decoupled weight decay via PyTorch's native `AdamW`. 2-D parameters (weight matrices, embeddings) receive decay; 1-D parameters (RMSNorm weights) do not. Tied weights are deduplicated automatically.

```bash
uv run python src/train.py --dataset data/shakespeare.txt --weight_decay 0.1
```

---

### Gradient clipping

```bash
# Default clip norm is 1.0; set 0 to disable
uv run python src/train.py --dataset data/shakespeare.txt --clip_norm 0.5
```

---

### Gradient accumulation

Effective batch size = `batch_size × accum_steps`.

```bash
# Effective batch = 4 × 8 = 32, same memory as batch_size=4
uv run python src/train.py --dataset data/shakespeare.txt \
    --batch_size 4 --accum_steps 8
```

---

### GQA — Grouped Query Attention

Set `--n_kv_head` to a divisor of `--n_head` to enable GQA. The KV cache shrinks by `n_head / n_kv_head` — critical for long contexts.

```bash
# 32 query heads, 8 KV heads — 4× smaller KV cache
uv run python src/train.py --dataset data/shakespeare.txt \
    --n_head 32 --n_kv_head 8 --n_embd 512 --n_layer 8
```

Or use a preset that has GQA built in:

```bash
uv run python src/train.py --dataset data/shakespeare.txt --model llama3-1b
```

---

### RoPE base (context extension)

`--rope_base` controls the frequency base θ for Rotary Position Embeddings. Higher values extend the effective context length without retraining.

```bash
# Default: 10000 (GPT-2/3 style)
# LLaMA-3 uses 500000 for its 128K context window
uv run python src/train.py --dataset data/shakespeare.txt \
    --rope_base 500000 --n_ctx 8192
```

---

### Sliding-window chunks

```bash
# 50% overlap — twice as many training chunks from the same text
uv run python src/train.py --dataset data/shakespeare.txt \
    --n_ctx 128 --stride 64
```

---

### Validation split

```bash
# Disable validation
uv run python src/train.py --dataset data/shakespeare.txt --val_fraction 0

# Custom split and eval frequency
uv run python src/train.py --dataset data/shakespeare.txt \
    --val_fraction 0.15 --eval_every 50
```

---

### Pre-tokenised binary

For large datasets, pre-tokenise once and skip re-tokenisation on every run:

```bash
# Tokenise once (produces data/books.npy)
uv run python src/pretokenize.py --input data/books/ --output data/books.npy

# Train from binary — fast startup, no tokenizer overhead
uv run python src/train.py --dataset_bin data/books.npy --n_vocab 50257 \
    --model llama3-8b --bf16 --grad_checkpoint
```

---

### Weights & Biases logging

```bash
uv run python src/train.py --dataset data/shakespeare.txt --wandb

# Custom project name
uv run python src/train.py --dataset data/shakespeare.txt \
    --wandb --wandb_project my-gpt-run
```

Logs `train_loss`, `val_loss`, and `lr` per step. Requires `wandb login` on first use.

---

### Resume training

Resumes automatically from the latest checkpoint when `--save_dir` contains one with a matching config:

```bash
# First run — stops at step 2000
uv run python src/train.py --dataset data/shakespeare.txt --steps 2000

# Continue from step 2000 — optimizer and LR schedule restored
uv run python src/train.py --dataset data/shakespeare.txt --steps 5000
```

Config mismatch → training starts fresh with a warning. Use a different `--save_dir` to keep both:

```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --model llama3-1b --save_dir checkpoints/llama3-1b
```

---

### Training on multiple files

```bash
uv run python src/train.py --dataset data/my_books/
```

---

### Override individual architecture parameters

```bash
# LLaMA-3 small with shorter context window
uv run python src/train.py --dataset data/shakespeare.txt \
    --model llama3-1b --n_ctx 1024

# Fully custom — GQA + longer context
uv run python src/train.py --dataset data/shakespeare.txt \
    --n_layer 8 --n_head 16 --n_kv_head 4 --n_embd 512 --n_ctx 2048
```

---

### Full argument reference

| Argument | Default | Description |
|---|---|---|
| `--dataset` | — | Text file or directory of `.txt`/`.md` files |
| `--dataset_bin` | — | Pre-tokenised `.npy` binary (mutually exclusive with `--dataset`) |
| `--model` | `small-cpu` | Model preset (see `--list_models`) |
| `--save_dir` | `checkpoints` | Directory for checkpoints, tokenizer, and config |
| `--tokenizer` | `char` | `char` or `tiktoken` |
| `--tiktoken_encoding` | `gpt2` | `gpt2` \| `cl100k_base` \| `o200k_base` |
| `--n_layer` | preset | Override transformer layers |
| `--n_head` | preset | Override query attention heads |
| `--n_kv_head` | preset | Override KV heads for GQA (default: same as n_head) |
| `--n_embd` | preset | Override embedding dimension |
| `--n_ctx` | preset | Override context window length |
| `--n_vocab` | — | Required when using `--dataset_bin` |
| `--dropout` | `0.0` | Dropout probability |
| `--rope_base` | `10000.0` | RoPE θ — increase for longer contexts |
| `--stride` | `n_ctx` | Sliding-window stride |
| `--val_fraction` | `0.1` | Validation fraction (0 to disable) |
| `--steps` | `500` | Max training steps (0 = no limit) |
| `--min_loss` | — | Stop when train loss drops below this value |
| `--batch_size` | `2` | Training batch size |
| `--learning_rate` | `1e-3` | Peak learning rate |
| `--warmup_steps` | `100` | Linear LR warmup steps |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--clip_norm` | `1.0` | Global gradient clip norm (0 to disable) |
| `--accum_steps` | `1` | Gradient accumulation micro-steps |
| `--bf16` | off | BF16 mixed precision (Ampere+ GPUs, recommended) |
| `--fp16` | off | FP16 mixed precision (older CUDA GPUs) |
| `--compile` | off | `torch.compile` the model (~2× throughput, Linux/CUDA) |
| `--grad_checkpoint` | off | Gradient checkpointing (~10× memory saving) |
| `--save_every` | `100` | Save checkpoint every N steps |
| `--eval_every` | `save_every` | Evaluate val loss every N steps |
| `--sample_every` | `50` | Print a generated sample every N steps |
| `--sample_length` | `200` | Tokens to generate per sample |
| `--seed` | `42` | Random seed |
| `--wandb` | off | Enable W&B logging |
| `--wandb_project` | `pocketmodel` | W&B project name |

---

## Chat

```bash
uv run python src/chat.py
```

### Chat options

```bash
uv run python src/chat.py \
    --checkpoints_dir checkpoints \
    --length 300 \
    --temperature 0.8 \
    --top_k 10
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoints_dir` | `checkpoints` | Directory with saved model and tokenizer |
| `--length` | `200` | Tokens to generate per response |
| `--temperature` | `0.8` | Higher = more creative, lower = more focused |
| `--top_k` | `10` | Sample from top-k most likely tokens (0 = no truncation) |

### Live commands

| Command | Example | Effect |
|---|---|---|
| `/temp <value>` | `/temp 0.5` | Adjust temperature on the fly |
| `/topk <value>` | `/topk 40` | Adjust top-k sampling |
| `/length <value>` | `/length 500` | Change response length |
| `/quit` | `/quit` | Exit |

---

## Example Training Runs

**Fast CPU test (~2 min):**
```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --model tiny --steps 500
```

**Recommended first run (CPU):**
```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --steps 0 --min_loss 1.2
```

**tiktoken + sliding window (better quality):**
```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --tokenizer tiktoken --stride 64 --steps 0 --min_loss 1.0
```

**Single GPU — gpt2-small:**
```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --model gpt2-small \
    --tokenizer tiktoken \
    --batch_size 4 --accum_steps 8 \
    --bf16 --compile --grad_checkpoint \
    --steps 0 --min_loss 0.9 \
    --save_dir checkpoints/gpt2-small
```

**Multi-GPU cloud run — llama3-8b:**
```bash
accelerate launch src/train.py \
    --dataset data/ \
    --model llama3-8b \
    --tokenizer tiktoken \
    --batch_size 4 --accum_steps 16 \
    --bf16 --compile --grad_checkpoint \
    --learning_rate 3e-4 --warmup_steps 2000 \
    --rope_base 500000 \
    --steps 0 --min_loss 1.0 \
    --wandb --wandb_project llama3-8b-run \
    --save_dir checkpoints/llama3-8b
```

---

## Project Structure

```
pocketmodel/
├── src/
│   ├── train.py        # Training — PyTorch + Accelerate + wandb
│   ├── chat.py         # Interactive chat — streaming KV-cache inference
│   ├── model.py        # GPT: RoPE, GQA, SwiGLU, RMSNorm, grad checkpoint
│   ├── configs.py      # Model presets (tiny → LLaMA-3 70B) + param estimator
│   ├── tokenizer.py    # CharTokenizer + TiktokenTokenizer + load factory
│   └── dataset.py      # Text loading, train/val split, sliding-window chunks
├── data/
│   └── shakespeare.txt  # Example dataset
├── checkpoints/          # Created during training
│   ├── ckpt_0000500.pt  # PyTorch state dict (model + optimizer + scheduler)
│   ├── config.json      # Pydantic GPTConfig — reloaded for resume and chat
│   └── tokenizer.json   # Tokenizer metadata
└── pyproject.toml
```

---

## Architecture Notes

Decoder-only transformer — same family as LLaMA-3, Mistral, Claude:

**RoPE (Rotary Position Embeddings)**
No learned position table (`wpe` removed). Frequencies are precomputed and applied directly to Q and K inside each attention layer via complex-number rotation. The buffer is 4× n_ctx so the model can be extended beyond its training context without recomputation.

**GQA (Grouped-Query Attention)**
Q has `n_head` heads; K and V have `n_kv_head` heads (fewer). Before the attention dot product, K and V are tiled via `repeat_interleave` to match `n_head`. The KV cache stores the compact `n_kv_head` version — at `n_kv_head=8`, `n_head=32`, the cache is 4× smaller than full MHA.

**SwiGLU MLP**
`output = down(silu(gate(x)) × up(x))`. Three weight matrices instead of two. Hidden dimension is `8/3 × n_embd` rounded to the nearest 64 or 256 — this keeps total parameter count equal to a standard 4× GELU MLP. Gate and up projections run in parallel, the Hadamard product with SiLU is the gating mechanism.

**RMSNorm**
`x × rsqrt(mean(x²) + ε) × weight`. No mean centering, no bias — 20–30% faster than LayerNorm. Applied before both attention and MLP (pre-norm residual), plus a final norm before the logit projection.

**KV Cache**
Each block returns its `(k, v)` tensors as `present_kv`. Subsequent tokens concatenate to the cache rather than recomputing. RoPE angles are baked into the cached keys at the positions they were computed — incremental tokens only rotate their own new keys. Trim by slicing `[:, :, 1:, :]` when the cache exceeds `n_ctx`.

**Gradient checkpointing**
`torch.utils.checkpoint` with `use_reentrant=False` wraps each block's forward pass during training. Activations are discarded after the forward pass and recomputed during backward — trading ~30% extra compute for removing the O(n_layer × seq_len) activation memory from GPU.

**Weight tying**
`lm_head.weight = transformer.wte.weight` — the output projection shares the token embedding matrix. Saves `n_vocab × n_embd` parameters (~38M for a 50K-vocab model with n_embd=768). `configure_optimizers` deduplicates by tensor identity so the shared weight is only in one param group.

**Residual projection scaling**
`o_proj` and `down_proj` are initialised with std `0.02 / sqrt(2 × n_layer)` — keeps the residual stream variance stable at initialisation regardless of depth (GPT-2 paper, reused in LLaMA).

---

## Library Stack

| Library | Version | Role |
|---|---|---|
| `torch` | ≥ 2.3 | Core framework — autograd, modules, Flash Attention via SDPA, `torch.compile` |
| `numpy` | ≥ 2.0 | Data chunking and sampling |
| `tiktoken` | ≥ 0.7 | Byte-level BPE tokenizer (GPT-2/4 vocab) |
| `accelerate` | ≥ 0.30 | Multi-GPU, BF16/FP16 mixed precision, gradient sync |
| `wandb` | ≥ 0.17 | Experiment tracking (optional) |
| `pydantic` | ≥ 2.7 | Typed, validated `GPTConfig` with JSON serialisation |
| `flash-attn` | ≥ 2.5 | *(optional)* Explicit FA2 for sequences > 4K or H100 kernels |
