import torch.nn as nn

from input_utils.mixup_utils import Mixup


class MixupAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """mixup and cutmix augmentation, does nothing if both has alpha 0"""
        super().__init__()
        self.args = args
        self.config = args.dataset_config["mixup"]
        self.config["num_classes"] = args.dataset_config[args.task]["num_classes"]
        self.mixup_func = Mixup(**args.dataset_config["mixup"])

        if "regression" in args.task:
            raise Exception("Mixup is not supported for regression task.")

    def forward(self, org_loc_inputs, labels=None):
        """
        Fake forward function of the no miss modality generator.
        x: [b, c, i, s]
        Return: Same shape as x, 1 means available, 0 means missing.
        """
        # TODO: Contrastive learning mixup, mixup function with no labels
        aug_loc_inputs, aug_labels = self.mixup_func(org_loc_inputs, labels, self.args.dataset_config)

        return aug_loc_inputs, None, aug_labels
