import torch
from torch.nn import Module


class ZScoreNorm(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor):
        with torch.no_grad():
            mean = torch.mean(x)
            std = torch.std(x)
            x = (x - mean) / std
            return x


class Unsqueeze(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.unsqueeze(self.dim)
            return x


class Squeeze(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor):
        with torch.no_grad():
            x = x.squeeze()
            return x
