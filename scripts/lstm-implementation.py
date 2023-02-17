import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCellNew(nn.Module):
    def __init__(self, input_sz, hidden_sz, bias=True):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.randn((input_sz, hidden_sz * 4)))
        self.weight_hh = nn.Parameter(torch.randn((hidden_sz, hidden_sz * 4)))
        self.bias_ih = nn.Parameter(torch.zeros(hidden_sz * 4))
        self.bias_hh = nn.Parameter(torch.zeros(hidden_sz * 4))

    def forward(self, x, h, c):
        # T x B x hidden_sz
        out = (
            x @ self.weight_ih
            + h @ self.weight_hh
            + self.bias_ih
            + self.bias_hh
        )
        i, f, g, o = torch.split(out, 100, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_t = f * c + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class LSTMNew(nn.Module):
    def __init__(self, input_sz, hidden_sz, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_sz = hidden_sz
        self.cells = nn.ModuleList(
            [
                LSTMCellNew(input_sz, hidden_sz)
                if i == 0
                else LSTMCellNew(hidden_sz, hidden_sz)
                for i in range(self.num_layers)
            ]
        )

    def forward(self, x, h_t, c_t):
        # x  :      T     x B x hidden_sz
        # h_t: num_layers x B x hidden_sz
        # c_t: num_layers x B x hidden_sz
        T, B, _ = x.shape
        H = torch.zeros(T, B, self.hidden_sz)
        for i, cell in enumerate(self.cells):
            h, c = h_t[i], c_t[i]
            if i > 0:
                x = H
            for t in range(T):
                h, c = cell(x[t], h, c)
                H[t] = h
            # last hidden state for each layer
            h_t[i], c_t[i] = h, c
        # Truncated BPTT
        return H, (h_t.detach(), c_t.detach())
