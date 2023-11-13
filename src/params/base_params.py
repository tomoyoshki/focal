import os
import argparse
import numpy as np

from params.output_paths import set_model_weight_file, set_output_paths, set_model_weight_folder
from params.params_util import *
from input_utils.yaml_utils import load_yaml


def parse_base_args(option="train"):
    """
    Parse the args.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-tag",
        type=str,
        default=None,
        help="The tag of execution, for record only.",
    )

    # dataset config
    parser.add_argument(
        "-dataset",
        type=str,
        default="MOD",
        help="Dataset to evaluate.",
    )

    parser.add_argument(
        "-task",
        type=str,
        default=None,
        help="The downstream task to evaluate.",
    )

    parser.add_argument(
        "-model",
        type=str,
        default="SW_Transformer",
        help="The backbone classification model to use.",
    )

    parser.add_argument(
        "-learn_framework",
        type=str,
        default="no",
        help="Which framework to use",
    )

    parser.add_argument(
        "-stage",
        type=str,
        default="pretrain",
        help="The pretrain/finetune, used for foundation model only.",
    )

    parser.add_argument(
        "-label_ratio",
        type=float,
        default=1.0,
        help="Only used in supervised training or finetune stage, specify the ratio of labeled data.",
    )

    parser.add_argument(
        "-model_weight",
        type=str,
        default=None,
        help="Specify the model weight path to evaluate.",
    )

    parser.add_argument(
        "-batch_size",
        type=int,
        default=None,
        help="Specify the batch size for training.",
    )

    parser.add_argument(
        "-gpu",
        type=str,
        default="0",
        help="Specify which GPU to use.",
    )

    args = parser.parse_args()

    args.option = option

    return args
