"""
Model preset configurations.

All presets define (n_layer, n_head, n_kv_head, n_embd, n_ctx).
Constraints:
  - n_embd must be divisible by n_head
  - n_head must be divisible by n_kv_head

Architecture:
  - GPT-2 style presets  : n_kv_head == n_head  (full MHA, original GPT-2)
  - LLaMA-3 style presets: n_kv_head  < n_head  (GQA — smaller KV cache)

Usage:
    python src/train.py --list_models
    python src/train.py --dataset data/text.txt --model llama3-8b
"""

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

MODEL_PRESETS = {
    # ── CPU-friendly (tiny experiments) ─────────────────────────────────
    "tiny":          dict(n_layer=2,  n_head=2,  n_kv_head=2,  n_embd=64,    n_ctx=128),
    "small-cpu":     dict(n_layer=4,  n_head=4,  n_kv_head=4,  n_embd=128,   n_ctx=128),

    # ── GPT-2 family — full MHA, context 1024 ───────────────────────────
    "gpt2-small":    dict(n_layer=12, n_head=12, n_kv_head=12, n_embd=768,   n_ctx=1024),
    "gpt2-medium":   dict(n_layer=24, n_head=16, n_kv_head=16, n_embd=1024,  n_ctx=1024),
    "gpt2-large":    dict(n_layer=36, n_head=20, n_kv_head=20, n_embd=1280,  n_ctx=1024),
    "gpt2-xl":       dict(n_layer=48, n_head=25, n_kv_head=25, n_embd=1600,  n_ctx=1024),

    # ── GPT-3 family — full MHA, context 2048 ───────────────────────────
    "gpt3-small":    dict(n_layer=12, n_head=12, n_kv_head=12, n_embd=768,   n_ctx=2048),
    "gpt3-medium":   dict(n_layer=24, n_head=16, n_kv_head=16, n_embd=1024,  n_ctx=2048),
    "gpt3-xl":       dict(n_layer=24, n_head=16, n_kv_head=16, n_embd=2048,  n_ctx=2048),
    "gpt3-6.7b":     dict(n_layer=32, n_head=32, n_kv_head=32, n_embd=4096,  n_ctx=2048),
    "gpt3-13b":      dict(n_layer=40, n_head=40, n_kv_head=40, n_embd=5120,  n_ctx=2048),

    # ── LLaMA-3 family — GQA + 8K context ───────────────────────────────
    # n_kv_head=8 across all sizes: 4–8× smaller KV cache than full MHA
    "llama3-1b":     dict(n_layer=16, n_head=32, n_kv_head=8,  n_embd=2048,  n_ctx=8192),
    "llama3-3b":     dict(n_layer=28, n_head=24, n_kv_head=8,  n_embd=3072,  n_ctx=8192),
    "llama3-8b":     dict(n_layer=32, n_head=32, n_kv_head=8,  n_embd=4096,  n_ctx=8192),
    "llama3-70b":    dict(n_layer=80, n_head=64, n_kv_head=8,  n_embd=8192,  n_ctx=8192),

    # ── Mistral-7B — GQA + sliding window, 32K context ──────────────────
    "mistral-7b":    dict(n_layer=32, n_head=32, n_kv_head=8,  n_embd=4096,  n_ctx=32768),
}


def get_preset(name: str) -> dict:
    """Return a copy of the preset dict, raising a clear error on unknown names."""
    if name not in MODEL_PRESETS:
        known = ', '.join(MODEL_PRESETS)
        raise ValueError(f"Unknown model preset '{name}'. Available: {known}")
    return dict(MODEL_PRESETS[name])


# ---------------------------------------------------------------------------
# Parameter estimation (updated for new architecture)
# ---------------------------------------------------------------------------

def estimate_params(n_vocab: int, n_ctx: int, n_embd: int,
                    n_head: int, n_layer: int,
                    n_kv_head: int | None = None,
                    ffn_hidden: int | None = None) -> int:
    """
    Approximate parameter count for the modern architecture:
      - No position embedding table (RoPE — no learned params)
      - GQA: Q heads = n_head, KV heads = n_kv_head
      - SwiGLU: 3 matrices × (n_embd × ffn_hidden) instead of 2 × (n_embd × 4*n_embd)
      - RMSNorm: weight only (no bias), 1 × n_embd per norm
      - No bias in Linear layers
    """
    if n_kv_head is None:
        n_kv_head = n_head
    head_dim = n_embd // n_head

    # Token embedding (no wpe — RoPE is parameter-free)
    embeddings = n_vocab * n_embd

    # Attention per block
    attn = (
        n_head    * head_dim * n_embd +     # Q
        n_kv_head * head_dim * n_embd * 2 + # K + V
        n_head    * head_dim * n_embd        # O
    )

    # SwiGLU MLP per block
    if ffn_hidden is None:
        raw  = int(8 / 3 * n_embd)
        mult = 64 if n_embd < 512 else 256
        ffn_hidden = ((raw + mult - 1) // mult) * mult
    mlp = 3 * n_embd * ffn_hidden           # gate + up + down

    # RMSNorm: ln_1 + ln_2 per block, ln_f final
    norms = (2 * n_layer + 1) * n_embd

    return embeddings + n_layer * (attn + mlp) + norms


def format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f'{n / 1_000_000_000:.2f} B'
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f} M'
    return f'{n:,}'


def list_presets() -> str:
    header = (f"{'Preset':<16} {'n_layer':>7} {'n_head':>7} {'n_kv':>5} "
              f"{'n_embd':>7} {'n_ctx':>7}  {'~Params (vocab=50k)':>20}")
    sep = '-' * 78
    rows = [header, sep]
    for name, cfg in MODEL_PRESETS.items():
        approx = estimate_params(n_vocab=50_000, **cfg)
        gqa    = '*' if cfg['n_kv_head'] < cfg['n_head'] else ' '
        rows.append(
            f"{name:<16} {cfg['n_layer']:>7} {cfg['n_head']:>7} "
            f"{cfg['n_kv_head']:>4}{gqa} {cfg['n_embd']:>7} {cfg['n_ctx']:>7}  "
            f"{format_params(approx):>20}"
        )
    rows.append('\n* = GQA (n_kv_head < n_head) — smaller KV cache')
    return '\n'.join(rows)
