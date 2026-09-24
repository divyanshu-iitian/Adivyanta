"""A small, randomly initialized decoder-only Transformer."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 128
    n_layer: int = 5
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc_out = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, t, 3, self.n_head, c // self.n_head).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(b, t, c))
        return x + self.fc_out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids, labels=None):
        b, t = ids.shape
        if t > self.cfg.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.cfg.block_size}")
        x = self.token(ids) + self.pos(torch.arange(t, device=ids.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100) if labels is not None else None
        return logits, loss

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def config_dict(self):
        return asdict(self.cfg)
