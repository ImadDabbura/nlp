import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNCellNew(nn.Module):
    def __init__(self, input_sz, hidden_sz, bias=True):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.randn((input_sz, hidden_sz)))
        self.weight_hh = nn.Parameter(torch.randn((hidden_sz, hidden_sz)))
        self.bias_ih = nn.Parameter(torch.zeros(hidden_sz))
        self.bias_hh = nn.Parameter(torch.zeros(hidden_sz))

    def forward(self, x, h, c):
        # B x hidden_sz
        return torch.tanh(
            x @ self.weight_ih
            + h @ self.weight_hh
            + self.bias_ih
            + self.bias_hh
        )


class RNNNew(nn.Module):
    def __init__(self, input_sz, hidden_sz, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_sz = hidden_sz
        self.cells = nn.ModuleList(
            [
                RNNCellNew(input_sz, hidden_sz)
                if i == 0
                else RNNCellNew(hidden_sz, hidden_sz)
                for i in range(self.num_layers)
            ]
        )

    def forward(self, x, h_t):
        # x  :      T     x B x hidden_sz
        # h_t: num_layers x B x hidden_sz
        T, B, _ = x.shape
        H = torch.zeros(T, B, self.hidden_sz)
        for i, cell in enumerate(self.cells):
            h = h_t[i]
            if i > 0:
                x = H
            for t in range(T):
                h = cell(x[t], h)
                H[t] = h
            # last hidden state for each layer
            h_t[i] = h
        # Truncated BPTT
        return H, h_t.detach()
