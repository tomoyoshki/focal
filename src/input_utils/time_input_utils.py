import os
import json
import torch

from tqdm import tqdm


def fft_preprocess(time_input, args):
    """Run FFT on the time-domain input.
    time_input: [b, c, i, s]
    freq_output: [b, c, i, s]
    """
    freq_output = dict()

    for loc in time_input:
        freq_output[loc] = dict()
        for mod in time_input[loc]:
            loc_mod_freq_output = torch.fft.fft(time_input[loc][mod], dim=-1)
            loc_mod_freq_output = torch.view_as_real(loc_mod_freq_output)
            loc_mod_freq_output = loc_mod_freq_output.permute(0, 1, 4, 2, 3)
            b, c1, c2, i, s = loc_mod_freq_output.shape
            loc_mod_freq_output = loc_mod_freq_output.reshape(b, c1 * c2, i, s)
            freq_output[loc][mod] = loc_mod_freq_output

    return freq_output