# pocketmodel

Train a GPT-style language model entirely from your own text — no pre-trained weights, no cloud, no external vocab files. Scales from a tiny CPU model up to GPT-2 / GPT-3 architectures.

Built on **PyTorch 2.x**, **Accelerate**, **tiktoken**, and **Pydantic**.

---

## Features

- **Two tokenizer backends** — character-level (zero deps, trains from your data) or tiktoken byte-level BPE (same vocab as GPT-2/3/4, 10× faster encoding)
- **Flash Attention** — `F.scaled_dot_product_attention` dispatches to FA2 kernels on CUDA automatically; manual fallback for CPU / older PyTorch
- **KV-cache inference** — O(1) per token in chat instead of O(n²); tokens stream as they generate
- **Sliding-window chunks** — overlapping context windows for more training signal per token
- **Train / validation split** — held-out eval loss tracked throughout training
- **Linear warmup + cosine LR decay** — standard schedule used by GPT-2/3
- **AdamW** — PyTorch native, decoupled weight decay; biases and LayerNorm params excluded automatically
- **Gradient clipping** — global norm clip prevents explosive updates
- **Gradient accumulation** — simulate large batches without extra memory
- **Mixed-precision training** — fp16 on CUDA via Accelerate
- **Multi-GPU training** — zero code changes; just run with `accelerate launch`
- **Residual projection scaling** — `c_proj` weights initialised at `0.02 / sqrt(2 × n_layer)` per the GPT-2 paper
- **Weight tying** — token embedding and output projection share the same matrix
- **Resume training** — automatic checkpoint detection with config compatibility check
- **Weights & Biases** — optional experiment tracking with `--wandb`
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

Any `.txt` file works — just point `--dataset` at it.

### 2. Train

```bash
uv run python src/train.py --dataset data/shakespeare.txt
```

### 3. Chat

```bash
uv run python src/chat.py
```

Type a prompt and the model streams its continuation. Type `/quit` to exit.

```
You › To be, or not to be

Model › that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows...
```

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
# tiktoken with GPT-4 encoding
uv run python src/train.py --dataset data/shakespeare.txt \
    --tokenizer tiktoken --tiktoken_encoding cl100k_base
```

**When to use which:**

| Dataset size | Recommendation |
|---|---|
| < 500 KB | `char` — simple, no overhead |
| 500 KB – 10 MB | Either; tiktoken trains faster |
| > 10 MB | `tiktoken` — far more efficient, richer token semantics |

---

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

### Learning rate schedule

Linear warmup + cosine decay is enabled by default. LR rises from 0 to `--learning_rate` over `--warmup_steps` steps, then decays via cosine to 10% of the peak.

```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --learning_rate 1e-3 --warmup_steps 200
```

---

### AdamW (weight decay)

Decoupled weight decay applied via PyTorch's native `AdamW`. Biases and LayerNorm parameters are automatically excluded from decay.

```bash
# Increase weight decay to reduce overfitting on small datasets
uv run python src/train.py --dataset data/shakespeare.txt --weight_decay 0.05
```

---

### Gradient clipping

```bash
# Default clip norm is 1.0; set 0 to disable
uv run python src/train.py --dataset data/shakespeare.txt --clip_norm 0.5
```

---

### Gradient accumulation

Accumulate gradients over N micro-batches before one optimizer step. Effective batch size = `batch_size × accum_steps`.

```bash
# Effective batch = 2 × 4 = 8, same memory as batch_size=2
uv run python src/train.py --dataset data/shakespeare.txt \
    --batch_size 2 --accum_steps 4
```

---

### Sliding-window chunks

By default chunks are non-overlapping (`stride = n_ctx`). A smaller stride exposes more sequence boundaries and is useful on small datasets.

```bash
# 50% overlap — twice as many training chunks from the same text
uv run python src/train.py --dataset data/shakespeare.txt \
    --n_ctx 128 --stride 64
```

---

### Validation split

10% of tokens are held out by default. Validation loss is printed every `--eval_every` steps.

```bash
# Disable validation
uv run python src/train.py --dataset data/shakespeare.txt --val_fraction 0

# Custom split and eval frequency
uv run python src/train.py --dataset data/shakespeare.txt \
    --val_fraction 0.15 --eval_every 50
```

---

### Mixed precision (GPU only)

```bash
uv run python src/train.py --dataset data/shakespeare.txt --fp16
```

Requires a CUDA GPU. Falls back gracefully on CPU.

---

### Multi-GPU training

No code changes required — just launch with `accelerate`:

```bash
# Uses all visible GPUs
accelerate launch src/train.py --dataset data/shakespeare.txt --fp16

# Configure which GPUs, precision, etc.
accelerate config   # interactive setup
accelerate launch src/train.py --dataset data/shakespeare.txt
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

Training resumes automatically from the latest checkpoint when `--save_dir` already contains one with a matching architecture:

```bash
# First run
uv run python src/train.py --dataset data/shakespeare.txt --steps 2000

# Continue from step 2000 — optimizer and LR schedule state restored
uv run python src/train.py --dataset data/shakespeare.txt --steps 5000
```

If the architecture changes, use a different `--save_dir`:

```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --model gpt2-small --save_dir checkpoints/gpt2-small
```

---

### Training on multiple files

Point `--dataset` at a directory — all `.txt` and `.md` files are loaded and concatenated:

```bash
uv run python src/train.py --dataset data/my_books/
```

---

### Override individual architecture parameters

Any `--n_*` flag overrides the corresponding preset value:

```bash
# GPT-3 small with a shorter context window
uv run python src/train.py --dataset data/shakespeare.txt \
    --model gpt3-small --n_ctx 256

# Fully custom architecture
uv run python src/train.py --dataset data/shakespeare.txt \
    --n_layer 6 --n_head 8 --n_embd 256 --n_ctx 512
```

---

### Full argument reference

| Argument | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Path to a `.txt` file or a directory of `.txt`/`.md` files |
| `--model` | `small-cpu` | Model preset (see `--list_models`) |
| `--save_dir` | `checkpoints` | Directory for checkpoints, tokenizer, and config |
| `--tokenizer` | `char` | `char` or `tiktoken` |
| `--tiktoken_encoding` | `gpt2` | tiktoken encoding: `gpt2` \| `cl100k_base` \| `o200k_base` |
| `--n_layer` | preset | Override transformer layers |
| `--n_head` | preset | Override attention heads |
| `--n_embd` | preset | Override embedding dimension (must be divisible by `n_head`) |
| `--n_ctx` | preset | Override context window length |
| `--dropout` | `0.0` | Dropout probability (useful to reduce overfitting on small datasets) |
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
| `--fp16` | off | Enable mixed-precision training (CUDA only) |
| `--save_every` | `100` | Save a checkpoint every N steps |
| `--eval_every` | `save_every` | Evaluate val loss every N steps |
| `--sample_every` | `50` | Print a generated sample every N steps |
| `--sample_length` | `200` | Tokens to generate per sample |
| `--seed` | `42` | Random seed |
| `--wandb` | off | Enable Weights & Biases logging |
| `--wandb_project` | `pocketmodel` | W&B project name |

---

## Chat

```bash
uv run python src/chat.py
```

Tokens are streamed to the terminal as they are generated. The KV cache is reused across tokens, so generation stays fast regardless of response length.

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
    --model small-cpu --steps 0 --min_loss 1.2
```

**tiktoken + sliding window (better quality, same size):**
```bash
uv run python src/train.py --dataset data/shakespeare.txt \
    --tokenizer tiktoken \
    --stride 64 --steps 0 --min_loss 1.0
```

**Serious training (GPU recommended):**
```bash
accelerate launch src/train.py --dataset data/shakespeare.txt \
    --model gpt2-small \
    --tokenizer tiktoken \
    --batch_size 8 --accum_steps 4 \
    --steps 0 --min_loss 0.9 \
    --fp16 --wandb \
    --save_dir checkpoints/gpt2-small
```

---

## Project Structure

```
pocketmodel/
├── src/
│   ├── train.py       # Training script — PyTorch + Accelerate + wandb
│   ├── chat.py        # Interactive chat — streaming KV-cache inference
│   ├── model.py       # GPT transformer (Pydantic config, Flash Attention, KV cache)
│   ├── configs.py     # Model size presets (tiny → GPT-3 175B)
│   ├── tokenizer.py   # CharTokenizer and TiktokenTokenizer with load factory
│   └── dataset.py     # Text loading, train/val split, sliding-window chunking
├── data/
│   └── shakespeare.txt  # Example dataset
├── checkpoints/         # Saved model weights (created during training)
│   ├── ckpt_0000500.pt  # PyTorch checkpoint (model + optimizer + scheduler)
│   ├── config.json      # Pydantic-serialised GPTConfig
│   └── tokenizer.json   # Tokenizer metadata
└── pyproject.toml
```

---

## Architecture Notes

Decoder-only transformer (same family as GPT-2/3) implemented in PyTorch 2.x:

- **Token + position embeddings** — learned tables `wte` and `wpe`; weights tied to the output projection (no extra params for the LM head)
- **N transformer blocks** — each: LayerNorm → multi-head causal self-attention → residual, then LayerNorm → MLP (GELU tanh-approx, 4× hidden) → residual
- **Final LayerNorm** → logit projection (weight-tied to `wte`)
- **Flash Attention** — `F.scaled_dot_product_attention` (PyTorch 2.0+) automatically uses Flash Attention 2 kernels on CUDA; safe fallback to manual attention with a registered causal mask on CPU or older PyTorch
- **Causal mask** — lower-triangular; during incremental KV-cache decode the query (length 1) can attend to all cached keys without masking
- **Residual projection init** — `c_proj` in both attention and MLP initialised with std `0.02 / sqrt(2 × n_layer)`, keeping residual stream variance stable at initialisation regardless of depth (GPT-2 paper)
- **KV cache** — during inference, each block returns its `(k, v)` tensors; subsequent tokens concatenate to the cache rather than recomputing, giving O(1) generation per token
- **AdamW param groups** — 2-D parameters (weight matrices, embeddings) receive weight decay; 1-D parameters (biases, LayerNorm scale/bias) do not; tied weights are deduplicated to avoid double-counting
- **Accelerate integration** — `Accelerator` handles device placement, fp16 loss scaling, and gradient sync across GPUs with no changes to the training logic

---

## Library Stack

| Library | Role |
|---|---|
| `torch >= 2.3` | Core framework — autograd, modules, Flash Attention via SDPA |
| `numpy >= 2.0` | Data chunking and sampling |
| `tiktoken >= 0.7` | Byte-level BPE tokenizer (GPT-2/4 vocab) |
| `accelerate >= 0.30` | Multi-GPU, mixed precision, gradient sync |
| `wandb >= 0.17` | Experiment tracking (optional) |
| `pydantic >= 2.7` | Typed, validated model config with JSON serialisation |
| `flash-attn >= 2.5` | *(optional)* Explicit FA2 for sequences > 4K or advanced kernels |
