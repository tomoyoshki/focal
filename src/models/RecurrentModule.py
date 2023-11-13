import torch
import torch.nn as nn


class RecurrentBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers=2, dropout_ratio=0) -> None:
        """The initialization of the recurrent block."""
        super().__init__()

        self.gru = nn.GRU(
            in_channel, out_channel, num_layers, bias=True, batch_first=True, dropout=dropout_ratio, bidirectional=True
        )

    def forward(self, x):
        """The forward function of the recurrent block.
        TODO: Add mask such that only valid intervals are considered in taking the mean.
        Args:
            x (_type_): [b, c_in, intervals]
        Output:
            [b, c_out]
        """
        # [b, c, i] --> [b, i, c]
        x = x.permute(0, 2, 1)

        # GRU --> mean
        # [b, i, c] --> [b, i, c]
        output, hidden_output = self.gru(x)

        # [b, i, c] --> [b, c]
        output = torch.mean(output, dim=1)

        return output, hidden_output