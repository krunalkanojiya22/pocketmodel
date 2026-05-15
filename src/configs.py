"""
Model preset configurations spanning GPT-2 to GPT-3 scale.

All presets define (n_layer, n_head, n_embd, n_ctx).
Constraint: n_embd must be divisible by n_head.

Usage in train.py:
    python src/train.py --dataset data/sample.txt --model gpt2-small
    python src/train.py --dataset data/sample.txt --model gpt3-medium --n_ctx 512
"""

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

MODEL_PRESETS = {
    # ── Local / CPU-friendly ─────────────────────────────────────────────
    "tiny":         dict(n_layer=2,  n_head=2,  n_embd=64,    n_ctx=128),
    "small-cpu":    dict(n_layer=4,  n_head=4,  n_embd=128,   n_ctx=128),

    # ── GPT-2 family (OpenAI, 2019) — context window 1024 ───────────────
    "gpt2-small":   dict(n_layer=12, n_head=12, n_embd=768,   n_ctx=1024),  # 117 M
    "gpt2-medium":  dict(n_layer=24, n_head=16, n_embd=1024,  n_ctx=1024),  # 345 M
    "gpt2-large":   dict(n_layer=36, n_head=20, n_embd=1280,  n_ctx=1024),  # 774 M
    "gpt2-xl":      dict(n_layer=48, n_head=25, n_embd=1600,  n_ctx=1024),  # 1.5 B

    # ── GPT-3 family (OpenAI, 2020) — context window 2048 ───────────────
    "gpt3-small":   dict(n_layer=12, n_head=12, n_embd=768,   n_ctx=2048),  # 125 M
    "gpt3-medium":  dict(n_layer=24, n_head=16, n_embd=1024,  n_ctx=2048),  # 350 M
    "gpt3-large":   dict(n_layer=24, n_head=16, n_embd=1536,  n_ctx=2048),  # 760 M
    "gpt3-xl":      dict(n_layer=24, n_head=16, n_embd=2048,  n_ctx=2048),  # 1.3 B
    "gpt3-2.7b":    dict(n_layer=32, n_head=32, n_embd=2560,  n_ctx=2048),  # 2.7 B
    "gpt3-6.7b":    dict(n_layer=32, n_head=32, n_embd=4096,  n_ctx=2048),  # 6.7 B
    "gpt3-13b":     dict(n_layer=40, n_head=40, n_embd=5120,  n_ctx=2048),  # 13 B
    "gpt3-175b":    dict(n_layer=96, n_head=96, n_embd=12288, n_ctx=2048),  # 175 B
}


def get_preset(name: str) -> dict:
    """Return a copy of the preset dict, raising a clear error on unknown names."""
    if name not in MODEL_PRESETS:
        known = ", ".join(MODEL_PRESETS)
        raise ValueError(f"Unknown model preset '{name}'. Available: {known}")
    return dict(MODEL_PRESETS[name])


def estimate_params(n_vocab: int, n_ctx: int, n_embd: int,
                    n_head: int, n_layer: int) -> int:
    """
    Approximate total trainable parameter count for a GPT model.

    Breakdown:
      - Token embedding (wte):      n_vocab * n_embd
      - Position embedding (wpe):   n_ctx   * n_embd
      - Per transformer block:
          LayerNorm x2:             4 * n_embd
          Attention QKV + proj:     4 * n_embd^2 + 4 * n_embd
          MLP fc + proj:            8 * n_embd^2 + 5 * n_embd
      - Final LayerNorm:            2 * n_embd
      - LM head shares wte (no extra params)
    """
    embeddings   = (n_vocab + n_ctx) * n_embd
    per_block    = 12 * n_embd ** 2 + 13 * n_embd
    final_ln     = 2 * n_embd
    return embeddings + n_layer * per_block + final_ln


def format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M"
    return f"{n:,}"


def list_presets() -> str:
    lines = [f"{'Preset':<16} {'n_layer':>7} {'n_head':>7} {'n_embd':>7} {'n_ctx':>6}  {'~Params (vocab=50k)':>20}"]
    lines.append("-" * 70)
    for name, cfg in MODEL_PRESETS.items():
        approx = estimate_params(
            n_vocab=50_000, **cfg
        )
        lines.append(
            f"{name:<16} {cfg['n_layer']:>7} {cfg['n_head']:>7} "
            f"{cfg['n_embd']:>7} {cfg['n_ctx']:>6}  {format_params(approx):>20}"
        )
    return "\n".join(lines)
