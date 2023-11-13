from params.base_params import parse_base_args
from params.params_util import set_auto_params


def parse_train_params():
    """Parse the training params."""
    args = parse_base_args("train")
    args = set_auto_params(args)
    return args
