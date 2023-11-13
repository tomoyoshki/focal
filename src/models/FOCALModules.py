import torch.nn as nn


class FOCAL(nn.Module):
    def __init__(self, args, backbone):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(FOCAL, self).__init__()

        self.args = args
        self.config = args.dataset_config["FOCAL"]
        self.backbone_config = args.dataset_config[args.model]
        self.modalities = args.dataset_config["modality_names"]

        # build encoders
        self.backbone = backbone

    def forward(self, aug_freq_input1, aug_freq_input2, proj_head=False):
        """
        Input:
            freq_input1: Input of the first augmentation.
            freq_input2: Input of the second augmentation.
        Output:
            mod_features1: Projected mod features of the first augmentation.
            mod_features2: Projected mod features of the second augmentation.
        """
        # compute features
        mod_features1 = self.backbone(aug_freq_input1, class_head=False, proj_head=proj_head)
        mod_features2 = self.backbone(aug_freq_input2, class_head=False, proj_head=proj_head)

        return mod_features1, mod_features2


def split_features(mod_features):
    """
    Split the feature into private space and shared space.
    mod_feature: [b, seq, dim], where we use the sequence sampler
    """
    split_mod_features = {}

    for mod in mod_features:
        if mod_features[mod].ndim == 2:
            split_dim = mod_features[mod].shape[1] // 2
            split_mod_features[mod] = {
                "shared": mod_features[mod][:, 0:split_dim],
                "private": mod_features[mod][:, split_dim:],
            }
        else:
            b, seq, dim = mod_features[mod].shape
            split_dim = dim // 2
            split_mod_features[mod] = {
                "shared": mod_features[mod][:, :, 0:split_dim],
                "private": mod_features[mod][:, :, split_dim : 2 * split_dim],
            }

    return split_mod_features
