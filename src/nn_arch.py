from typing import Sequence
import torch.nn as nn
import torch

class GlobalSmileNetVector(nn.Module):
    def __init__(self, in_dim: int = 14, hidden: Sequence[int]=(250,), out_dim: int = 10):
        super().__init__()
        assert len(hidden) == 1, "paper: single hidden layer"
        h = int(hidden[0])

        self.fc1 = nn.Linear(in_dim, h)
        self.act  = nn.Softplus()
        self.head = nn.Linear(h, out_dim)

        nn.init.xavier_uniform_(self.fc1.weight)   
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.head.weight)           
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.act(self.fc1(x)))
