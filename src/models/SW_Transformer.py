import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoderLayer
from input_utils.padding_utils import get_padded_size
from models.FusionModules import TransformerFusionBlock
from timm.models.layers import trunc_normal_

from models.SwinModules import (
    BasicLayer,
    PatchEmbed,
    PatchMerging,
)


class SW_Transformer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["SW_Transformer"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.num_segments = args.dataset_config["num_segments"]

        self.drop_rate = self.config["dropout_ratio"]
        self.norm_layer = nn.LayerNorm

        self.init_encoder()

    def init_encoder(self) -> None:

        self.freq_interval_layers = nn.ModuleDict()
        self.patch_embed = nn.ModuleDict()
        self.absolute_pos_embed = nn.ModuleDict()
        self.mod_patch_embed = nn.ModuleDict()
        self.mod_in_layers = nn.ModuleDict()

        self.layer_dims = {}
        self.img_sizes = {}

        for loc in self.locations:
            self.freq_interval_layers[loc] = nn.ModuleDict()
            self.patch_embed[loc] = nn.ModuleDict()
            self.absolute_pos_embed[loc] = nn.ParameterDict()
            self.mod_in_layers[loc] = nn.ModuleDict()

            self.layer_dims[loc] = {}
            self.img_sizes[loc] = {}
            for mod in self.modalities:
                # Decide the spatial size for "image"
                stride = self.config["in_stride"][mod]
                spectrum_len = self.args.dataset_config["loc_mod_spectrum_len"][loc][mod]
                img_size = (self.num_segments, spectrum_len // stride)

                # get the padded image size
                padded_img_size = get_padded_size(
                    img_size,
                    self.config["window_size"][mod],
                    self.config["patch_size"]["freq"][mod],
                    len(self.config["time_freq_block_num"][mod]),
                )
                self.img_sizes[loc][mod] = padded_img_size

                # Patch embedding and Linear embedding (H, W, in_channel) -> (H / p_size, W / p_size, C)
                self.patch_embed[loc][mod] = PatchEmbed(
                    img_size=padded_img_size,
                    patch_size=self.config["patch_size"]["freq"][mod],
                    in_chans=self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod] * stride,
                    embed_dim=self.config["time_freq_out_channels"],
                    norm_layer=self.norm_layer,
                )
                patches_resolution = self.patch_embed[loc][mod].patches_resolution

                # Absolute positional embedding (optional)
                self.absolute_pos_embed[loc][mod] = nn.Parameter(
                    torch.zeros(1, self.patch_embed[loc][mod].num_patches, self.config["time_freq_out_channels"])
                )
                trunc_normal_(self.absolute_pos_embed[loc][mod], std=0.02)

                # Swin Transformer Block
                self.freq_interval_layers[loc][mod] = nn.ModuleList()

                # Drop path rate
                dpr = [
                    x.item()
                    for x in torch.linspace(
                        0, self.config["drop_path_rate"], sum(self.config["time_freq_block_num"][mod])
                    )
                ]  # stochastic depth decay rule

                for i_layer, block_num in enumerate(
                    self.config["time_freq_block_num"][mod]
                ):  # different downsample ratios
                    down_ratio = 2**i_layer
                    layer_dim = int(self.config["time_freq_out_channels"] * down_ratio)
                    layer = BasicLayer(
                        dim=layer_dim, 
                        input_resolution=(
                            patches_resolution[0] // down_ratio,
                            patches_resolution[1] // down_ratio,
                        ),
                        num_heads=self.config["time_freq_head_num"],
                        window_size=self.config["window_size"][mod].copy(),
                        depth=block_num,
                        drop=self.drop_rate,
                        attn_drop=self.config["attn_drop_rate"],
                        drop_path=dpr[
                            sum(self.config["time_freq_block_num"][mod][:i_layer]) : sum(
                                self.config["time_freq_block_num"][mod][: i_layer + 1]
                            )
                        ],
                        norm_layer=self.norm_layer,
                        downsample=PatchMerging
                        if (i_layer < len(self.config["time_freq_block_num"][mod]) - 1)
                        else None,
                    )
                    self.freq_interval_layers[loc][mod].append(layer)

                # Unify the input channels for each modality
                self.mod_in_layers[loc][mod] = nn.Linear(
                    (patches_resolution[0] // down_ratio) * (patches_resolution[1] // down_ratio) * layer_dim,
                    self.config["loc_out_channels"],
                )

        # Loc fusion, [b, i, c], loc contextual feature extraction + loc fusion
        if len(self.locations) > 1:
            self.loc_context_layers = nn.ModuleDict()
            self.loc_fusion_layer = nn.ModuleDict()
            for mod in self.modalities:
                """Single mod contextual feature extraction"""
                module_list = [
                    TransformerEncoderLayer(
                        d_model=self.config["loc_out_channels"],
                        nhead=self.config["loc_head_num"],
                        dim_feedforward=self.config["loc_out_channels"],
                        dropout=self.config["dropout_ratio"],
                        batch_first=True,
                    )
                    for _ in range(self.config["loc_block_num"])
                ]
                self.loc_context_layers[mod] = nn.Sequential(*module_list)

                """Loc fusion layer for each mod"""
                self.loc_fusion_layer[mod] = TransformerFusionBlock(
                    self.config["loc_out_channels"],
                    self.config["loc_head_num"],
                    self.config["dropout_ratio"],
                    self.config["dropout_ratio"],
                )

        # mod fusion layer
        """Mod feature projection, and attention fusion."""
        out_dim = self.args.dataset_config["FOCAL"]["emb_dim"]
        self.mod_projectors = nn.ModuleDict()
        for mod in self.modalities:
            self.mod_projectors[mod] = nn.Sequential(
                nn.Linear(self.config["loc_out_channels"], out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
        self.mod_fusion_layers = TransformerFusionBlock(
            self.config["loc_out_channels"],
            self.config["loc_head_num"],
            self.config["dropout_ratio"],
            self.config["dropout_ratio"],
        )
        self.sample_dim = self.config["loc_out_channels"]

        # Classification layer
        if self.args.train_mode == "supervised" or self.config["pretrained_head"] == "linear":
            """Linear classification layers for supervised learning or finetuning."""
            self.class_layer = nn.Sequential(
                nn.Linear(self.sample_dim, self.args.dataset_config[self.args.task]["num_classes"]),
            )
        else:
            """Non-linear classification layers for self-supervised learning."""
            self.class_layer = nn.Sequential(
                nn.Linear(self.sample_dim, self.config["fc_dim"]),
                nn.GELU(),
                nn.Linear(self.config["fc_dim"], self.args.dataset_config[self.args.task]["num_classes"]),
            )

    def pad_input(self, freq_x, loc, mod):
        stride = self.config["in_stride"][mod]
        spectrum_len = self.args.dataset_config["loc_mod_spectrum_len"][loc][mod]
        img_size = (self.num_segments, spectrum_len // stride)
        freq_input = freq_x[loc][mod]

        # [b, c, i, spectrum] -- > [b, i, spectrum, c]
        freq_input = torch.permute(freq_input, [0, 2, 3, 1])
        b, i, s, c = freq_input.shape

        # Forces both audio and seismic to have the same "img" size
        freq_input = torch.reshape(freq_input, (b, i, s // stride, c * stride))

        # Repermute back to [b, c, i, spectrum], (b, c, h, w) required in PatchEmbed
        freq_input = torch.permute(freq_input, [0, 3, 1, 2])

        # Pad [i, spectrum] to the required padding size
        padded_img_size = self.patch_embed[loc][mod].img_size
        padded_height = padded_img_size[0] - img_size[0]
        padded_width = padded_img_size[1] - img_size[1]

        # test different padding
        freq_input = F.pad(input=freq_input, pad=(0, padded_width, 0, padded_height), mode="constant", value=0)

        return freq_input, padded_img_size

    def forward_encoder(self, patched_input, class_head=True, proj_head=False):
        """
        If class_head is False, we return the modality features; otherwise, we return the classification results.
        time-freq feature extraction --> loc fusion --> mod concatenation --> class layer
        """
        # Step 1: Feature extractions on time interval (i) and spectrum (s) domains
        mod_loc_features = {mod: [] for mod in self.modalities}
        for loc in self.locations:
            for mod in self.modalities:
                embeded_input = patched_input[loc][mod]
                b = embeded_input.shape[0]

                # Absolute positional embedding
                if self.config["APE"]:
                    embeded_input = embeded_input + self.absolute_pos_embed[loc][mod]

                # SwinTransformer Layer block
                for layer in self.freq_interval_layers[loc][mod]:
                    freq_interval_output = layer(embeded_input)
                    embeded_input = freq_interval_output

                # Unify the input channels for each modality
                freq_interval_output = self.mod_in_layers[loc][mod](freq_interval_output.reshape([b, -1]))
                freq_interval_output = freq_interval_output.reshape(b, 1, -1)

                # Append the modality feature to the list
                mod_loc_features[mod].append(freq_interval_output)

        # Concatenate the location features, [b, i, location, c]
        for mod in self.modalities:
            mod_loc_features[mod] = torch.stack(mod_loc_features[mod], dim=2)

        # Step 2: Loc feature fusion and extraction for each mod, [b, i, location, c]
        mod_features = []
        for mod in mod_loc_features:
            if len(self.locations) > 1:
                """Extract mod feature with peer-feature context"""
                b, i, locs, c = mod_loc_features[mod].shape
                mod_loc_input = mod_loc_features[mod].reshape([b * i, locs, c])
                mod_loc_context_feature = self.loc_context_layers[mod](mod_loc_input)
                mod_loc_context_feature = mod_loc_context_feature.reshape([b, i, locs, c])

                """Mod feature fusion, [b, 1, 1, c] -- > [b, c]"""
                mod_feature = self.loc_fusion_layer[mod](mod_loc_context_feature)
                mod_feature = mod_feature.flatten(start_dim=1)
                mod_features.append(mod_feature)
            else:
                mod_features.append(mod_loc_features[mod].flatten(start_dim=1))

        # Step 3: Mod concatenation, [b, 1, mod, c]
        if not class_head:
            """Perform mod feature projection."""
            if proj_head:
                sample_features = {}
                for i, mod in enumerate(self.modalities):
                    sample_features[mod] = self.mod_projectors[mod](mod_features[i])
                return sample_features
            else:
                return dict(zip(self.modalities, mod_features))
        else:
            """Mod feature projection and attention-based fusion"""
            mod_features = torch.stack(mod_features, dim=1)
            mod_features = mod_features.unsqueeze(dim=1)
            sample_features = self.mod_fusion_layers(mod_features).flatten(start_dim=1)

            logits = self.class_layer(sample_features)
            return logits

    def patch_forward(self, freq_x):
        """Patch the input and mask for pretrianing
        """
        embeded_inputs = {}
        for loc in self.locations:
            embeded_inputs[loc] = {}
            for mod in self.modalities:
                # Pad the input and store the padded input
                freq_input, _ = self.pad_input(freq_x, loc, mod)

                # Patch Partition and Linear Embedding
                embeded_input = self.patch_embed[loc][mod](freq_input)

                embeded_inputs[loc][mod] = embeded_input
        return embeded_inputs

    def forward(self, freq_x, class_head=True, proj_head=False):
        # PatchEmbed the input
        patched_inputs = self.patch_forward(freq_x)

        if class_head:
            """Finetuning the classifier"""
            logits = self.forward_encoder(patched_inputs, class_head)
            return logits
        else:
            enc_mod_features = self.forward_encoder(patched_inputs, class_head, proj_head)
            return enc_mod_features
