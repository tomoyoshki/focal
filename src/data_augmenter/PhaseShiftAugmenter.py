import torch
import torch.nn as nn

from random import random, randint
from input_utils.mixup_utils import Mixup
from math import floor


class PhaseShiftAugmenter(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["phase_shift"]
        self.p = self.config["prob"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]

    def forward(self, org_loc_inputs, labels):
        """
        Frequency masking augmentation with a random frequency band.
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

                    # real --> complex
                    b, c, i, s = mod_input.shape
                    mod_input = mod_input.reshape([b, c // 2, 2, i, s])
                    mod_input = mod_input.permute([0, 1, 3, 4, 2]).contiguous()
                    mod_input = torch.view_as_complex(mod_input)

                    # perform the random shift
                    random_angle = (random() - 0.5) * 2 * torch.pi
                    new_angles = mod_input.angle() + random_angle
                    new_reals = mod_input.abs() * torch.cos(new_angles)
                    new_imags = mod_input.abs() * torch.sin(new_angles)
                    aug_mod_input = torch.stack([new_reals, new_imags], dim=-1)

                    # complex --> real
                    aug_mod_input = aug_mod_input.permute([0, 1, 4, 2, 3])
                    b, c1, c2, i, s = aug_mod_input.shape
                    aug_mod_input = aug_mod_input.reshape([b, c1 * c2, i, s])

                    # store the augmented input
                    aug_loc_inputs[loc][mod] = aug_mod_input
                    aug_mod_labels.append(1)
                else:
                    aug_loc_inputs[loc][mod] = org_loc_inputs[loc][mod]
                    aug_mod_labels.append(0)

        aug_mod_labels = torch.Tensor(aug_mod_labels).to(self.args.device)
        aug_mod_labels = aug_mod_labels.unsqueeze(0).tile([b, 1]).float()

        return aug_loc_inputs, aug_mod_labels, labels
