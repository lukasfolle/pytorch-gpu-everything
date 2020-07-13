import torch
from torch.nn import Module


class ZScoreNorm(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor):
        mean = torch.mean(x)
        std = torch.std(x)
        return (x - mean) / std
