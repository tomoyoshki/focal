from params.base_params import parse_base_args
from params.params_util import set_auto_params


def parse_test_params():
    """Parse the testing params."""
    args = parse_base_args("test")
    args = set_auto_params(args)
    return args
