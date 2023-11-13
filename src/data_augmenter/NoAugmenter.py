import torch
import torch.nn as nn
import numpy as np


class NoAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """None missing modality generator"""
        super().__init__()
        self.args = args

    def forward(self, loc_inputs, labels=None):
        """
        Fake forward function of the no miss modality generator.
        x: loc --> mod --> [b, c, i, s]
        Return: Same shape as x, 1 means available, 0 means missing.
        """
        return loc_inputs, None, labels
