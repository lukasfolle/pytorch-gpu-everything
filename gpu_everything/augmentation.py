from abc import ABC
import torch
from torch.nn import Module


class Augmentation(Module):
    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    def random_prob_reached(self):
        if self.prob > torch.rand(size=(1,)):
            return True
        return False


class RandomFlip(Augmentation):
    def __init__(self, dims: tuple, prob: float):
        super().__init__(prob)
        self.dims = dims

    def forward(self, x):
        if self.random_prob_reached():
            return torch.flip(x, dims=self.dims)
        return x


class RandomRotate90(Augmentation):
    def __init__(self, dims: tuple, prob: float, max_number_rot: int):
        super().__init__(prob)
        self.dims = dims
        self.max_number_rot = max_number_rot

    def forward(self, x):
        if self.random_prob_reached():
            num_rotations = int(torch.randint(low=1, high=self.max_number_rot + 1, size=(1,)))
            return torch.rot90(x, num_rotations, self.dims)
