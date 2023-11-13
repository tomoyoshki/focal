import torch
import torch.nn as nn

from random import random

from tsai.data.transforms import TSTimeWarp
from tsai.data.core import TSTensor


class TimeWarpAugmenter(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["time_warp"]
        self.p = self.config["prob"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.warp_func = TSTimeWarp(magnitude=self.config["magnitude"], order=self.config["order"])

    def forward(self, org_loc_inputs, labels=None):
        """
        Fake forward function of the scaling augmenter. Operate in the time domain.
        split_idx: 0 for training set and 1 for validation set.
        magnitude: the strength of the warping function.
        order: the number of knots in the warping.
        x: [b, c, i, s]
        Return: Same shape as x. A single random scaling factor for each (loc, mod).
        """
        aug_loc_inputs = {}
        aug_mod_labels = []
        b = None  # batch size

        for loc in self.locations:
            aug_loc_inputs[loc] = {}
            for mod in self.modalities:
                # retrieve the batch size first
                if b is None:
                    b = org_loc_inputs[loc][mod].shape[0]

                if random() < self.p:
                    mod_input = org_loc_inputs[loc][mod].clone()
                    b, c, i, s = mod_input.shape
                    mod_input = torch.reshape(mod_input, (b, c, i * s))
                    aug_loc_inputs[loc][mod] = self.warp_func(TSTensor(mod_input), split_idx=0).reshape(b, c, i, s).data
                    aug_mod_labels.append(1)
                else:
                    aug_loc_inputs[loc][mod] = org_loc_inputs[loc][mod]
                    aug_mod_labels.append(0)

        aug_mod_labels = torch.Tensor(aug_mod_labels).to(self.args.device)
        aug_mod_labels = aug_mod_labels.unsqueeze(0).tile([b, 1]).float()

        return aug_loc_inputs, aug_mod_labels, labels
