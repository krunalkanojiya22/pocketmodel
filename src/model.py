import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, model_validator


class GPTConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    n_vocab:  int   = 0
    n_ctx:    int   = 1024
    n_embd:   int   = 768
    n_head:   int   = 12
    n_layer:  int   = 12
    dropout:  float = 0.0

    @model_validator(mode='after')
    def _check_head_divisibility(self):
        if self.n_vocab > 0 and self.n_embd % self.n_head != 0:
            raise ValueError(
                f'n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})'
            )
        return self


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout

        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)

        # PyTorch >= 2.0 SDPA dispatches to Flash Attention on CUDA automatically.
        self._use_sdpa = hasattr(F, 'scaled_dot_product_attention')
        if not self._use_sdpa:
            # Fallback causal mask for older PyTorch builds.
            self.register_buffer(
                'bias',
                torch.tril(torch.ones(config.n_ctx, config.n_ctx))
                     .view(1, 1, config.n_ctx, config.n_ctx),
            )

    def forward(self, x: torch.Tensor, past_kv=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v)

        if self._use_sdpa:
            # is_causal=True only when processing the full prompt (no past cache):
            # query and key have the same length, so standard causal masking applies.
            # During incremental decode (past_kv is not None), T_q=1 can attend to
            # all cached keys — no masking needed.
            dp = self.dropout_p if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dp, is_causal=(past_kv is None)
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            T_q, T_k = q.size(2), k.size(2)
            att = att.masked_fill(
                self.bias[:, :, T_k - T_q:T_k, :T_k] == 0, float('-inf')
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), present_kv


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(F.gelu(self.c_fc(x), approximate='tanh')))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor, past_kv=None):
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.n_vocab, config.n_embd),
            wpe  = nn.Embedding(config.n_ctx,   config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.n_vocab, bias=False)

        # Weight tying: output projection shares the token embedding matrix.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # GPT-2 residual projection scaling: keeps residual stream variance
        # stable at init regardless of depth.
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets=None, past_kvs=None):
        B, T = idx.shape
        device = idx.device
        past_len = 0 if past_kvs is None else past_kvs[0][0].size(2)

        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )

        new_kvs = []
        for i, block in enumerate(self.transformer.h):
            x, kv = block(x, past_kv=(past_kvs[i] if past_kvs is not None else None))
            new_kvs.append(kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss, new_kvs

    @property
    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        n -= self.transformer.wte.weight.numel()  # tied weight counted once
        return n

    def configure_optimizers(self, lr: float, weight_decay: float):
        """AdamW: weight decay on matrices (dim >= 2), not on biases / LN params."""
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
