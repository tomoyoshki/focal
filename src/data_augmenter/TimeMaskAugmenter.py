import torch
import torch.nn as nn

from random import random, randint
from input_utils.mixup_utils import Mixup
from math import floor


class TimeMaskAugmenter(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["time_mask"]
        self.p =  self.config["prob"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.max_duration = floor(args.dataset_config["num_segments"] * self.config["mask_ratio"])

    def forward(self, org_loc_inputs, labels):
        """
        Time masking augmentation with a random time duration.
        Reference: https://towardsdatascience.com/data-augmentation-for-speech-recognition-e7c607482e78.
        x: [b, c, i, s]
        Return: Same shape as x.
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
                    duration = randint(1, self.max_duration)
                    start_interval = torch.randint(0, mod_input.shape[2] - duration, (1,)).item()
                    mod_input[:, :, :, start_interval : start_interval + duration] = 0
                    aug_loc_inputs[loc][mod] = mod_input
                    aug_mod_labels.append(1)
                else:
                    aug_loc_inputs[loc][mod] = org_loc_inputs[loc][mod]
                    aug_mod_labels.append(0)

        aug_mod_labels = torch.Tensor(aug_mod_labels).to(self.args.device)
        aug_mod_labels = aug_mod_labels.unsqueeze(0).tile([b, 1]).float()

        return aug_loc_inputs, aug_mod_labels, labels
