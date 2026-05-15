# pocketmodel

Train a GPT-style language model entirely from your own text — no pre-trained weights, no external vocab files. Scales from a tiny CPU-friendly model up to GPT-2 / GPT-3 architectures.

---

## Features

- **Two tokenizer backends** — character-level (zero deps) or subword BPE via sentencepiece
- **Sliding-window chunks** — overlapping context windows for more training signal per token
- **Train / validation split** — held-out eval loss tracked throughout training
- **Linear warmup + cosine LR decay** — standard schedule used by GPT-2/3
- **AdamW** — decoupled weight decay; biases and LayerNorm params are excluded
- **Gradient clipping** — global norm clip prevents explosive updates
- **Gradient accumulation** — simulate large batches without extra memory
- **Mixed-precision training** — optional fp16 on supported GPUs
- **Residual projection scaling** — `c_proj` weights initialised at `0.02 / sqrt(2 × n_layer)` per the GPT-2 paper
- **KV-cache inference** — O(1) per token in chat instead of O(n²); tokens stream as they generate
- **Resume training** — automatic checkpoint detection with hparam compatibility check

---

## Requirements

- Python 3.10 or 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager

---

## Setup

```bash
git clone <your-repo-url>
cd pocketmodel

# Creates .venv and installs all dependencies (including sentencepiece)
uv sync
```

---

## Quick Start

### 1. Get a dataset

```bash
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
     -o data/shakespeare.txt
```

Any `.txt` file works — the model builds its vocabulary from the text.

### 2. Train

```bash
uv run python src/train.py --dataset data/shakespeare.txt
```

### 3. Chat

```bash
uv run python src/chat.py
```

Type a prompt and the model streams its continuation. Type `/quit` to exit.

---

## Training

### Basic usage

```bash
uv run python src/train.py --dataset data/shakespeare.txt
```

### Tokenizer

| Flag | Value | Description |
|---|---|---|
| `--tokenizer` | `char` *(default)* | Character-level; vocab built from training data |
| `--tokenizer` | `bpe` | Subword BPE via sentencepiece; learns word-piece merges |
| `--vocab_size` | `1000` *(default)* | Number of BPE pieces (ignored for `char`) |

```bash
# BPE with 800 subword pieces
uv run python src/train.py --dataset data/shakespeare.txt \
    --tokenizer bpe --vocab_size 800
```

BPE tokens carry more semantic content per step — the model learns faster on larger datasets. For tiny datasets (< 500 KB) the character tokenizer is usually fine.

### Model presets

```bash
uv run python src/train.py --list_models
```

```
Preset           n_layer  n_head  n_embd  n_ctx   ~Params (vocab=50k)
----------------------------------------------------------------------
tiny                   2       2      64    128                 3.3 M
small-cpu              4       4     128    128                 7.2 M   ← default
gpt2-small            12      12     768   1024               124.2 M
gpt2-medium           24      16    1024   1024               354.6 M
gpt2-large            36      20    1280   1024               773.7 M
gpt2-xl               48      25    1600   1024                1.56 B
gpt3-small            12      12     768   2048               125.0 M
gpt3-medium           24      16    1024   2048               355.6 M
gpt3-large            24      16    1536   2048               759.9 M
gpt3-xl               24      16    2048   2048                1.32 B
gpt3-2.7b             32      32    2560   2048                2.65 B
gpt3-6.7b             32      32    4096   2048                6.66 B
gpt3-13b              40      40    5120   2048               12.85 B
gpt3-175b             96      96   12288   2048              174.60 B
```

> **CPU training:** use `tiny` or `small-cpu`. Everything above `gpt2-small` needs a GPU.

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

Ctrl+C always saves a checkpoint before exiting.

#### Target loss values

| Loss | What it means |
|---|---|
| ~4.2 | Random guessing (65-char vocab) |
| ~2.5 | Basic letter/word frequency learned |
| ~1.5 | Recognising common words and patterns |
| ~1.2 | Readable text with consistent style |
| < 0.5 | Overfitting — model is memorising the data |

### Learning rate schedule

Warmup + cosine decay is enabled by default. After `--warmup_steps` steps the LR rises linearly from 0 to `--learning_rate`, then decays via cosine to 10% of the peak.

```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --learning_rate 1e-3 --warmup_steps 200
```

### AdamW (weight decay)

Decoupled weight decay is applied after every optimiser step. Biases and LayerNorm parameters are excluded.

```bash
# Increase weight decay to reduce overfitting on small datasets
uv run python src/train.py --dataset data/shakespeare.txt --weight_decay 0.05
```

### Gradient clipping

```bash
# Default clip norm is 1.0; set 0 to disable
uv run python src/train.py --dataset data/shakespeare.txt --clip_norm 0.5
```

### Gradient accumulation

Accumulate gradients over N micro-batches before one optimiser step. Effective batch size = `batch_size × accum_steps`.

```bash
# Effective batch = 2 × 4 = 8, same memory as batch_size=2
uv run python src/train.py --dataset data/shakespeare.txt \
    --batch_size 2 --accum_steps 4
```

### Sliding-window chunks

By default chunks are non-overlapping (`stride = n_ctx`). A smaller stride exposes more sequence boundaries to the model and is especially useful on small datasets.

```bash
# 50% overlap: twice as many training chunks from the same text
uv run python src/train.py --dataset data/shakespeare.txt \
    --n_ctx 128 --stride 64
```

### Validation split

10% of tokens are held out by default. Validation loss is printed every `--eval_every` steps (defaults to `--save_every`).

```bash
# Disable validation
uv run python src/train.py --dataset data/shakespeare.txt --val_fraction 0

# Custom split and eval frequency
uv run python src/train.py --dataset data/shakespeare.txt \
    --val_fraction 0.15 --eval_every 50
```

### Mixed precision (GPU only)

```bash
uv run python src/train.py --dataset data/shakespeare.txt --fp16
```

Requires a CUDA GPU and TensorFlow ≥ 2.4. Falls back to fp32 with a warning on unsupported hardware.

### Resume training

Training resumes automatically from the latest checkpoint when `--save_dir` already contains one with matching architecture:

```bash
# First run
uv run python src/train.py --dataset data/shakespeare.txt --steps 2000

# Continue from step 2000
uv run python src/train.py --dataset data/shakespeare.txt --steps 5000
```

If the architecture changes, use a different `--save_dir`:

```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --model gpt2-small --save_dir checkpoints/gpt2-small
```

### Training on multiple files

Point `--dataset` at a directory — all `.txt` and `.md` files are loaded and concatenated:

```bash
uv run python src/train.py --dataset data/my_books/
```

### Override individual architecture parameters

Any `--n_*` flag overrides the corresponding preset value:

```bash
# GPT-3 small architecture with a shorter context window
uv run python src/train.py --dataset data/shakespeare.txt \
    --model gpt3-small --n_ctx 256

# Fully custom architecture
uv run python src/train.py --dataset data/shakespeare.txt \
    --n_layer 6 --n_head 8 --n_embd 256 --n_ctx 512
```

### Full argument reference

| Argument | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Path to a `.txt` file or a directory of `.txt`/`.md` files |
| `--model` | `small-cpu` | Model preset (see `--list_models`) |
| `--save_dir` | `checkpoints` | Directory to save checkpoints and tokenizer |
| `--tokenizer` | `char` | `char` or `bpe` |
| `--vocab_size` | `1000` | BPE vocabulary size |
| `--n_layer` | preset | Override transformer layers |
| `--n_head` | preset | Override attention heads |
| `--n_embd` | preset | Override embedding dimension (must be divisible by `n_head`) |
| `--n_ctx` | preset | Override context window length |
| `--stride` | `n_ctx` | Sliding-window stride for chunks |
| `--val_fraction` | `0.1` | Fraction of tokens held out for validation (0 to disable) |
| `--steps` | `500` | Max training steps (`0` = no limit) |
| `--min_loss` | `None` | Stop early when loss drops below this value |
| `--batch_size` | `2` | Training batch size |
| `--learning_rate` | `1e-3` | Peak learning rate |
| `--warmup_steps` | `100` | Linear LR warmup steps |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--clip_norm` | `1.0` | Global gradient clip norm (0 to disable) |
| `--accum_steps` | `1` | Gradient accumulation micro-steps |
| `--fp16` | off | Enable mixed-precision training |
| `--save_every` | `100` | Save a checkpoint every N steps |
| `--eval_every` | `save_every` | Evaluate val loss every N steps |
| `--sample_every` | `50` | Print a generated sample every N steps |
| `--sample_length` | `200` | Tokens to generate per sample |
| `--seed` | `42` | Random seed |

---

## Chat

```bash
uv run python src/chat.py
```

Tokens are streamed to the terminal as they are generated. The KV cache is reused across tokens, so generation stays fast regardless of response length.

```
You › To be, or not to be

Model › that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows...
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
| `/temp <value>` | `/temp 0.5` | Lower = more focused output |
| `/topk <value>` | `/topk 40` | Higher = more varied word choices |
| `/length <value>` | `/length 500` | Longer response |
| `/quit` | `/quit` | Exit |

---

## Example Training Runs

**Fast CPU test (~2 min):**
```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --model tiny --steps 500
```

**Recommended first run:**
```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --model small-cpu --steps 0 --min_loss 1.2
```

**BPE with sliding window (better quality, same size):**
```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --tokenizer bpe --vocab_size 800 \
    --stride 64 --steps 0 --min_loss 1.0
```

**Serious training (GPU recommended):**
```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --model gpt2-small --steps 0 --min_loss 0.9 \
    --batch_size 8 --accum_steps 4 \
    --save_dir checkpoints/gpt2-small
```

---

## Project Structure

```
pocketmodel/
├── src/
│   ├── train.py       # Training script
│   ├── chat.py        # Interactive chat with streaming KV-cache inference
│   ├── model.py       # GPT transformer (attention, MLP, residual blocks)
│   ├── configs.py     # Model size presets (tiny → GPT-3 175B)
│   ├── tokenizer.py   # CharTokenizer and BPETokenizer with shared load factory
│   └── dataset.py     # Text loading, train/val split, sliding-window chunking
├── data/
│   └── shakespeare.txt  # Example dataset
├── checkpoints/         # Saved model weights (created during training)
└── pyproject.toml
```

---

## Architecture Notes

The model is a decoder-only transformer (same family as GPT-2/3) implemented in TensorFlow 1.x compatibility mode:

- **Token + position embeddings** — learned tables `wte` and `wpe`
- **N transformer blocks** — each: LayerNorm → multi-head causal self-attention → residual, then LayerNorm → MLP (GELU, 4× hidden) → residual
- **Final LayerNorm** → logit projection weight-tied to `wte`
- **Causal mask** — lower-triangular attention prevents attending to future tokens
- **Residual projection init** — `c_proj` in both attention and MLP is initialised with std `0.02 / sqrt(2 × n_layer)`, keeping residual stream variance stable at initialisation regardless of depth
- **KV cache** — during inference, key/value tensors from prior positions are stored and concatenated rather than recomputed, making generation O(1) per token
