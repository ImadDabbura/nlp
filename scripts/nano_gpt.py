import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

# Config
batch_sz = 16
block_sz = 32
emb_dim = 128
head_sz = 64
n_heads = 4
n_layers = 4
dropout = 0.0
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"


class BigramLM(nn.Module):
    def __init__(self, vocab_sz, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim)

    def forward(self, x, targets=None):
        logits = self.embedding(x)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.softmax(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            logits = self(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_char], dim=-1)
        return x


class HeadEfficient(nn.Module):
    def __init__(self, emb_dim, block_sz, head_sz, dropout) -> None:
        super().__init__()
        self.W = nn.Linear(emb_dim, head_sz * 3, bias=False)
        self.register_buffer(
            "mask",
            torch.triu(-float("inf") * torch.ones(block_sz, block_sz), 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k, q, v = torch.split(x @ self.W, 3, dim=-1)
        w = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
        w = F.softmax(w + self.mask, dim=-1)
        w = self.dropout(w)
        out = w @ v
        return out  # B x T x head_sz


class Head(nn.Module):
    def __init__(self, emb_dim, block_sz, head_sz, dropout) -> None:
        super().__init__()
        self.key = nn.Linear(emb_dim, head_sz, bias=False)
        self.query = nn.Linear(emb_dim, head_sz, bias=False)
        self.value = nn.Linear(emb_dim, head_sz, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_sz, block_sz))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        w = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
        w = w.masked_fill(self.tril == 0, float("inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        v = self.value(x)
        out = w @ v
        return out  # B x T x head_sz


class MultiHeadAttentionEfficient(nn.Module):
    def __init__(self, emb_dim, block_sz, n_heads, head_sz, dropout) -> None:
        super().__init__()
        self.W = nn.Linear(emb_dim, n_heads * head_sz, bias=False)
        self.register_buffer(
            "mask",
            torch.triu(-float("inf") * torch.ones(block_sz, block_sz), 1),
        )
        self.proj = nn.Linear(head_sz, head_sz)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, T, d = x.shape
        k, q, v = torch.split(x @ self.W, 3, dim=-1)
        k, q, v = [
            a.reshape(N, T, n_heads, head_sz // n_heads).transpose(1, 2)
            for a in (q, k, v)
        ]
        w = F.softmax(
            q @ k.transpose(-2, -1) / (head_sz // n_heads) ** 0.5 + self.mask
        )
        x = (w @ v).transpose(1, 2).reshape(N, T, d)
        x = self.proj(x)
        return self.dropout(x)  # B x T x head_sz


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, block_sz, n_heads, head_sz, dropout) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(emb_dim, block_sz, head_sz // n_heads, dropout)
                for _ in range(n_heads)
            ]
        )
        self.proj = nn.Linear(head_sz, head_sz)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)  # B x T x head_sz


class FFN(nn.Module):
    def __init__(self, n_heads, head_sz, dropout) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(head_sz, n_heads * head_sz),
            nn.ReLU(),
            nn.Linear(n_heads * head_sz, head_sz),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, emb_dim, block_sz, head_sz, n_heads, dropout) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            emb_dim, block_sz, n_heads, head_sz, dropout
        )
        self.ffn = FFN(n_heads, head_sz, dropout)
        self.layer_norm_1 = nn.LayerNorm(head_sz // n_heads)
        self.layer_norm_2 = nn.LayerNorm(head_sz // n_heads)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm_1(x))
        x = x + self.ffn(self.layer_norm_2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(
        self,
        block_sz,
        vocab_sz,
        emb_dim,
        head_sz,
        n_heads,
        dropout,
        n_layers=1,
    ) -> None:
        super().__init__()
        self._block_sz = block_sz
        self.token_embedding = nn.Embedding(vocab_sz, emb_dim)
        self.position_embedding = nn.Embedding(block_sz, emb_dim)
        self.blocks = nn.Sequential(
            *[
                Block(emb_dim, block_sz, head_sz, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(head_sz)
        self.lm_head = nn.Linear(head_sz, vocab_sz)

    def forward(self, x, targets=None):
        B, T = x.shape
        x = self.token_embedding(x) + self.position_embedding(torch.arange(T))
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            logits = self(x[:, -self.block_sz :])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_char], dim=-1)
        return x


def get_data():
    pass

def prepare_data(data):
    pass


def get_batch():
    pass


def train():
    model = NanoGPT(block_sz, vocab_sz, emb_dim, head_sz, dropout)


if __name__ == "__main__":
    train()
