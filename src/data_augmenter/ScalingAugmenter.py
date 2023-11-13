import torch
import torch.nn as nn

from random import random


class ScalingAugmenter(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["scaling"]
        self.p = self.config["prob"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]

    def forward(self, org_loc_inputs, labels=None):
        """
        Fake forward function of the scaling augmenter. Operate in the time domain.
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
                    mod_input = org_loc_inputs[loc][mod]
                    scale_factor = torch.normal(1.0, self.config["std"], size=(1,)).to(self.args.device)
                    aug_loc_inputs[loc][mod] = mod_input * scale_factor
                    aug_mod_labels.append(1)
                else:
                    aug_loc_inputs[loc][mod] = org_loc_inputs[loc][mod]
                    aug_mod_labels.append(0)

        aug_mod_labels = torch.Tensor(aug_mod_labels).to(self.args.device)
        aug_mod_labels = aug_mod_labels.unsqueeze(0).tile([b, 1]).float()

        return aug_loc_inputs, aug_mod_labels, labels
