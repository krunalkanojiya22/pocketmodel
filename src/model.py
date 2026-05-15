"""
GPT model — modern frontier architecture.

Key upgrades over vanilla GPT-2:
  - RoPE (Rotary Position Embeddings) instead of learned wpe table
  - GQA (Grouped-Query Attention) — fewer KV heads reduces KV-cache size
  - SwiGLU MLP instead of GELU — better optimisation landscape
  - RMSNorm instead of LayerNorm — faster, equally stable
  - No bias in Linear layers — modern standard
  - Gradient checkpointing support — trade compute for memory
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from pydantic import BaseModel, ConfigDict, model_validator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class GPTConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    n_vocab:    int        = 0
    n_ctx:      int        = 2048
    n_embd:     int        = 768
    n_head:     int        = 12
    n_kv_head:  int        = 0        # 0 = same as n_head (full MHA); set < n_head for GQA
    n_layer:    int        = 12
    ffn_hidden: int | None = None     # SwiGLU hidden dim; auto-derived if None
    dropout:    float      = 0.0
    rope_base:  float      = 10000.0  # RoPE theta — raise for longer contexts

    @model_validator(mode='after')
    def _validate(self):
        if self.n_kv_head == 0:
            self.n_kv_head = self.n_head   # default: full MHA
        if self.n_vocab > 0:
            if self.n_embd % self.n_head != 0:
                raise ValueError(
                    f'n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})'
                )
            if self.n_head % self.n_kv_head != 0:
                raise ValueError(
                    f'n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})'
                )
        return self

    def get_ffn_hidden(self) -> int:
        """SwiGLU hidden dim: 8/3 × n_embd rounded to the nearest 64 or 256."""
        if self.ffn_hidden is not None:
            return self.ffn_hidden
        raw  = int(8 / 3 * self.n_embd)
        mult = 64 if self.n_embd < 512 else 256
        return ((raw + mult - 1) // mult) * mult


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation — faster than LayerNorm, no bias."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(head_dim: int, max_seq_len: int,
                          base: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex RoPE frequencies of shape (max_seq_len, head_dim // 2).

    Stored as a non-persistent buffer so it moves with the model device
    but is not saved in checkpoints (recomputed on load).
    """
    theta  = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t      = torch.arange(max_seq_len, dtype=torch.float32)
    freqs  = torch.outer(t, theta)                          # (T, head_dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)       # complex64


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               freqs_cis: torch.Tensor):
    """
    Rotate Q and K by their position-dependent angle.

    q/k:       (B, n_head,    T, head_dim)
    freqs_cis: (T, head_dim/2) complex  — already sliced to current positions
    """
    def rotate(x):
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        xc = xc * freqs_cis.unsqueeze(0).unsqueeze(0)      # broadcast over B, heads
        return torch.view_as_real(xc).flatten(3).type_as(x)
    return rotate(q), rotate(k)


# ---------------------------------------------------------------------------
# Attention — GQA + RoPE + Flash Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_groups  = config.n_head // config.n_kv_head  # GQA expansion factor
        self.head_dim  = config.n_embd // config.n_head
        self.dropout_p = config.dropout

        # Separate projections — K/V use n_kv_head instead of n_head (GQA)
        self.q_proj = nn.Linear(config.n_embd, config.n_head    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head  * self.head_dim, config.n_embd,   bias=False)

        # PyTorch >= 2.0 SDPA uses Flash Attention kernels on CUDA automatically
        self._use_sdpa = hasattr(F, 'scaled_dot_product_attention')
        if not self._use_sdpa:
            self.register_buffer(
                'bias',
                torch.tril(torch.ones(config.n_ctx, config.n_ctx))
                     .view(1, 1, config.n_ctx, config.n_ctx),
            )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, past_kv=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to the new positions only; past keys already have their angles baked in
        q, k = apply_rope(q, k, freqs_cis)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present_kv = (k, v)

        # GQA: tile KV heads to match Q heads before attention (no-op when n_groups == 1)
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        if self._use_sdpa:
            dp = self.dropout_p if self.training else 0.0
            y  = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dp, is_causal=(past_kv is None)
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att   = (q @ k.transpose(-2, -1)) * scale
            T_q, T_k = q.size(2), k.size(2)
            att   = att.masked_fill(
                self.bias[:, :, T_k - T_q:T_k, :T_k] == 0, float('-inf')
            )
            att = F.softmax(att, dim=-1)
            y   = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.o_proj(y), present_kv


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    """
    SwiGLU: output = down(silu(gate(x)) * up(x))

    Uses 8/3 × n_embd hidden units (vs 4× for GELU) — compensates for the
    extra gate projection while matching compute. Used by LLaMA, PaLM, Claude.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.get_ffn_hidden()
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.up_proj   = nn.Linear(config.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.drop      = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp  = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, past_kv=None):
        attn_out, present_kv = self.attn(self.ln_1(x), freqs_cis, past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config           = config
        self._grad_checkpoint = False

        head_dim = config.n_embd // config.n_head
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.n_vocab, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.n_vocab, bias=False)

        # Weight tying: output projection shares the token embedding matrix
        self.transformer.wte.weight = self.lm_head.weight

        # Precomputed RoPE frequencies — 4× n_ctx buffer allows context extension
        # without recomputation. Non-persistent: recomputed on load, not stored.
        self.register_buffer(
            'freqs_cis',
            precompute_rope_freqs(head_dim, config.n_ctx * 4, config.rope_base),
            persistent=False,
        )

        self.apply(self._init_weights)
        # Residual projection scaling: keeps gradient variance stable at depth
        for name, p in self.named_parameters():
            if name.endswith(('o_proj.weight', 'down_proj.weight')):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self):
        """Recompute activations during backward — trades 30% compute for ~10× memory."""
        self._grad_checkpoint = True

    def forward(self, idx: torch.Tensor, targets=None, past_kvs=None):
        B, T     = idx.shape
        device   = idx.device
        past_len = 0 if past_kvs is None else past_kvs[0][0].size(2)

        # Slice RoPE frequencies for the current positions
        freqs_cur = self.freqs_cis[past_len : past_len + T]

        x = self.transformer.drop(self.transformer.wte(idx))

        new_kvs = []
        for i, block in enumerate(self.transformer.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
            if self._grad_checkpoint and self.training and past_kv is None:
                # grad_checkpoint saves no activations; recomputes on backward.
                # Only safe when past_kv is None (always true during training).
                x, kv = grad_checkpoint(block, x, freqs_cur, use_reentrant=False)
            else:
                x, kv = block(x, freqs_cur, past_kv=past_kv)
            new_kvs.append(kv)

        x      = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return logits, loss, new_kvs

    @property
    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        n -= self.transformer.wte.weight.numel()    # tied weight, count once
        return n

    def configure_optimizers(self, lr: float, weight_decay: float):
        """AdamW: weight decay on matrices (dim >= 2), not on norms / scalars."""
        seen, decay, nodecay = set(), [], []
        for _, p in self.named_parameters():
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            (decay if p.dim() >= 2 else nodecay).append(p)
        return torch.optim.AdamW(
            [{'params': decay,   'weight_decay': weight_decay},
             {'params': nodecay, 'weight_decay': 0.0}],
            lr=lr, betas=(0.9, 0.95), eps=1e-8,
        )
