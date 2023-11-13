import torch
import torch.nn as nn
import numpy as np
from time import sleep
import math

import torch.nn.functional as F


class ConvTranspose2dPadding(torch.nn.ConvTranspose2d):
    """ConvTranspose2d that calculates same padding
    https://github.com/pytorch/pytorch/issues/67551,
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#ConvTranspose2d,
    """

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        if self.mode == "same":
            return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)
        else:
            return max((math.floor(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def set_padding(self, mode):
        """set padding to same or valid"""
        assert mode in {"same", "valid"}
        self.mode = mode

    def forward(self, x: torch.Tensor, output_size=None) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        self.padding = tuple([pad_h, pad_w])
        # self.output_padding = 2 * np.array(padding) - np.array(self.kernel_size) + np.array(self.stride)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, 2, self.dilation
        )

        return F.conv_transpose2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class ConvLayer2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        padding_mode,
        bias,
        dropout_ratio,
        activation="GELU",
    ) -> None:
        """The initialization of the 2D convolution layer.
        Structure: conv2d + batch_norm + relu + dropout
        Input shape:
        Single sensor: (b, c (2 * 3 or 1), intervals, spectral samples)
        Merge conv: (b, c (1), intervals, spectral samples)
        Activation: GELU > SILU ～ ELU > ReLU = Leaky RelU = PReLU
        """
        super().__init__()
        self.inc = in_channels
        self.out = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, track_running_stats=True)
        if activation == "GELU":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()

        """ p: Probability of an element to be zero-ed
           - Input: (N, C, H, W) or (C, H, W)
           - Output: (N, C, H, W) or (C, H, W)
        """
        self.dropout = nn.Dropout2d(p=dropout_ratio)

    def forward(self, x):
        """The forward function of the ConvLayer.

        InputShape: [b, c (1), INTERVAL, 2 * 3 * SPECTRAL_SAMPlES]
        OutputShape: [b, c_out, interval, spectral_samples]

        Args:
            x (_type_): _description_
        """
        conv_out = self.conv(x)
        conv_out = self.batch_norm(conv_out)
        conv_out = self.activation(conv_out)
        conv_out = self.dropout(conv_out)

        return conv_out


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        in_spectrum_len,
        interval_num=9,  # only needed when conv_len[0] > 0 and fuse time dimension inforamtion
        conv_lens=[[1, 3], [1, 3], [1, 3]],
        dropout_ratio=0,
        num_inter_layers=2,
        in_stride=1,
    ) -> None:
        """The initialization of the sensor convolution block.
        Structure: 3 * ConvLayer
        conv_lens gives the length of convolution kernel at each conv layer.
        At the input conv layer, necessary dimension unifying might be performed between acoustic and other sensors.
        """
        super().__init__()
        self.conv_lens = conv_lens
        self.num_inter_layers = num_inter_layers
        self.fuse_time_flag = self.conv_lens[1][0] > 1

        # define input conv layer
        self.conv_layer_in = ConvLayer2D(
            in_channels,
            int(out_channels / 2),
            kernel_size=conv_lens[0],
            stride=in_stride,
            padding="same" if np.max(in_stride) == 1 else "valid",
            padding_mode="zeros",
            bias=True,
            dropout_ratio=dropout_ratio,
        )

        # define inter conv layers with same input and output channels
        self.conv_layers_inter = nn.ModuleList()
        for i in range(self.num_inter_layers):
            conv_layer = ConvLayer2D(
                int(out_channels / 2),
                int(out_channels / 2),
                kernel_size=conv_lens[1],
                stride=1,
                padding="same",
                padding_mode="zeros",
                bias=True,
                dropout_ratio=dropout_ratio,
            )
            self.conv_layers_inter.append(conv_layer)

        # define output conv layer
        if self.fuse_time_flag:
            if in_stride == 1:
                last_in_channels = int(out_channels / 2 * in_spectrum_len * interval_num)
            else:
                last_in_channels = int(out_channels / 2 * in_spectrum_len * interval_num / in_stride[1])
        else:
            if in_stride == 1:
                last_in_channels = int(out_channels / 2 * in_spectrum_len)
            else:
                last_in_channels = int(out_channels / 2 * in_spectrum_len / in_stride[1])

        """No activation at the out conv layer."""
        self.conv_layer_out = nn.Conv1d(
            last_in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            padding_mode="zeros",
            bias=True,
        )

    def forward(self, x):
        """The forward function of the SensorConvBlock.

        Args:
            x (_type_): (b, c (2 * 3 or 1), i (intervals), s (spectrum))
        Output:
            [b, c, i]
        """

        # print("ConvBlock input: ", x.shape)
        # input conv layer
        conv_out = self.conv_layer_in(x)

        # print("ConvBlock conv_out: ", conv_out.shape)

        # inter conv layers
        for conv_layer in self.conv_layers_inter:
            conv_out = conv_out + conv_layer(conv_out)

        # reshape the output to (N, C_out, intervals)
        conv_out = conv_out.permute(0, 1, 3, 2)

        b, c, s, i = conv_out.shape
        if self.fuse_time_flag:
            conv_out = torch.reshape(conv_out, (b, c * s * i, 1))
        else:
            conv_out = torch.reshape(conv_out, (b, c * s, i))

        # map the dimension to the same as output, [b, c_out, i]
        conv_out = self.conv_layer_out(conv_out)

        return conv_out