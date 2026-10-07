"""A GPT-style language model, written from scratch in PyTorch.

The model reads a sequence of token ids and, for every position, predicts the
next token. It is a "decoder-only transformer":

    token ids
      -> token embedding + position embedding
      -> N x Block( self-attention , MLP )      each wrapped in a residual connection
      -> LayerNorm -> linear layer -> one score (logit) per vocabulary token

Self-attention lets every position look back at earlier positions; a causal
mask stops it from looking ahead, which would be cheating during training.
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 256  # the longest sequence the model can see (its context window)
    n_layer: int = 6  # number of transformer blocks
    n_head: int = 6  # attention heads per block
    n_embd: int = 192  # width of the model (size of each token's vector)
    dropout: float = 0.0


# Three sizes to compare in experiments. Roughly 1M, 3M and 7M parameters.
PRESETS = {
    "tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=128),
    "small": dict(n_layer=6, n_head=6, n_embd=192, block_size=256),
    "base": dict(n_layer=8, n_head=8, n_embd=256, block_size=256),
}


class CausalSelfAttention(nn.Module):
    """Each position gathers information from itself and earlier positions."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)  # query, key, value in one layer
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Lower-triangular matrix: position t may attend to positions 0..t only.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding width
        head_dim = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        # Split the width into heads: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # How strongly does each position (query) match each other position (key)?
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)  # (B, n_head, T, T)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # hide the future
        weights = self.attn_dropout(F.softmax(scores, dim=-1))  # each row sums to 1

        out = weights @ v  # weighted average of the values
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # glue the heads back together
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    """A small feed-forward network applied to every position independently."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    """One transformer block: attention then MLP, each added onto the input (residual)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)  # what each token means
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)  # where it is in the sequence
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the output layer reuses the token embedding matrix.
        # It saves parameters and usually improves small models.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Layers that write into the residual stream start smaller, so the
        # stream does not grow with depth at the start of training.
        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. Returns (logits, loss); loss is None without targets."""
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence of {T} tokens is longer than block_size {self.config.block_size}"
        positions = torch.arange(T, device=idx.device)
        x = self.dropout(self.tok_emb(idx) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy: how surprised the model is by the real next token.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 40, stop_token: int | None = None) -> torch.Tensor:
        """Continue the sequence `idx` (B, T) one token at a time."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size:]  # the model can only see block_size tokens
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # only the last position predicts the next token
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits[logits < kth] = float("-inf")  # keep only the k most likely tokens
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if stop_token is not None and idx.size(0) == 1 and next_id.item() == stop_token:
                break
        return idx

    def config_dict(self) -> dict:
        return asdict(self.config)


def pick_device() -> str:
    """Apple Silicon GPU if available, then NVIDIA GPU, otherwise the CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
