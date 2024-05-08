import torch
import torch.nn as nn
import torch.nn.functional as F

# config.head_dim is typically config.embed_dim / n_heads
# config.embed_dim is sometimes also called hidden_size
# Hidden size of the first layer in the FF NN is typically 4x the size of the embedding
# Most of the capacity and memorization is expected to happen in the first layer of the
# FF NN, which is what gets scaled when the model is scaled up
# FF NN uses 2 linear layers -> since the input has shape B x T x D, the linear layer is
# applied to each embedding vector independently in the sequence and batch, which leads to
# position-wise feedforward layer.

CONFIG = {
    "vocab_sz": 1000,
    "block_sz": 8,
    "intermediare_sz": None,
    "hidden_dropout_prob": "0.2",
    "num_attention_heads": 4,
    "hidden_sz": 64,
    "num_hidden_layers": 6,
    "embed_dim": 768,
    "num_classes": 2,
    "layer_norm_rps": 1e-12,
}


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_sz, config.embed_dim)
        self.position_embedding = nn.Embedding(
            config.block_sz, config.embed_dim
        )
        self.layer_norm = nn.LayerNorm(
            config.embed_dim, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout()

    def forward(self, x):
        # X is B x T
        # token_embeddings are B x T x config.embed_dim
        # position_embeddings are T x config.embed_dim
        embeddings = self.token_embedding(x) + self.position_embedding(
            torch.arange(x.shape[1])
        )
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class AttentionHead(nn.Module):
    def __init__(self, config, head_dim) -> None:
        super().__init__()
        self.k = nn.Linear(config.hidden_sz, head_dim, bias=False)
        self.q = nn.Linear(config.hidden_sz, head_dim, bias=False)
        self.v = nn.Linear(config.hidden_sz, head_dim, bias=False)
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.block_sz, config.block_sz))
        )

    def forward(self, x):
        # k,q,v are each B x T x config.hidden_sz
        k, q, v = [func(x) for func in (self.k, self.q, self.v)]
        # w is B x T x T
        w = q @ k.transpose(2, 1) / (k.shape[-1] ** 0.5)
        w = w.masked_fill(self.mask == 0, -float("inf"))
        w = F.softmax(w, dim=-1)
        # output is B x T x config.hidden_sz
        return w @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        head_dim = config.hidden_sz // config.num_attention_heads
        self.heads = nn.ModuleList(
            [
                AttentionHead(head_dim, config)
                for _ in range(config.num_attention_heads)
            ]
        )
        self.output = nn.Linear(config.hidden_sz, config.hidden_sz)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.output(x)


class FeedForwardNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # config.intermediate_sz is typically 4x hidden_sz
        self.l1 = nn.Linear(config.hidden_sz, config.intermediate_sz)
        self.l2 = nn.Linear(config.intermediate_sz, config.hidden_sz)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        return self.dropout(self.l2(F.gelu(self.l1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config.head_dim)
        self.layer_norm_2 = nn.LayerNorm(config.head_dim)
        self.ff = FeedForwardNN(config)

    def forward(self, x):
        # There are two arrangements for layer_norm:
        # Prelayer normalization & Postlayer normalization
        # we are using postlayer normalization arrangement
        x = self.layer_norm_1(x + self.attn(x))
        x = self.layer_norm_2(x + self.ff(x))
        # Prelayer normalization
        # x = self.layer_norm_1(x)
        # x = x + self.attn(x)
        # x = x + self.ff(self.layer_norm_2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder_layers = nn.Sequential(
            *[EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.head_dim, config.num_classes)

    def forward(self, x):
        # We take the hidden state of the [CLS] token as
        # input to the classifier
        x = self.encoder(x)[:, 0, :]
        x = self.dropout(x)
        return self.classifier(x)
