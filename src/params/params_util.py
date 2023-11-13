import os
import torch
import getpass

from params.output_paths import set_model_weight_file, set_output_paths, set_model_weight_folder
from input_utils.yaml_utils import load_yaml


def get_username():
    # The function to automatically get the username
    username = getpass.getuser()
    return username


def str_to_bool(flag):
    # Convert the string flag to bool.
    return True if flag.lower().strip() == "true" else False


def select_device(device="", batch_size=0, newline=True):
    # automatically select GPU if available else CPU
    s = f"Torch-{torch.__version__} "
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)"  # bytes to MB
        arg = f"cuda:0"
    elif mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += "MPS"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    print(s)

    return torch.device(arg)


def get_train_mode(learn_framework):
    """
    Automatically set the train mode according to the learn_framework.
    NOTE: Add the learn framework to this register when adding a new learn framework.
    """
    learn_framework_register = {
        "FOCAL": "contrastive",
        "no": "supervised",
    }

    if learn_framework in learn_framework_register:
        train_mode = learn_framework_register[learn_framework]
    else:
        raise ValueError(f"Invalid learn_framework provided: {learn_framework}")

    return train_mode


def set_task(args):
    """
    Set the default task according to the dataset.
    """
    task_default_task = {
        "ACIDS": "vehicle_classification",
        "MOD": "vehicle_classification",
        "RealWorld_HAR": "activity_classification",
        "PAMAP2": "activity_classification",
    }

    task = task_default_task[args.dataset] if args.task is None else args.task
    return task


def set_batch_size(args):
    """
    Automatically set the batch size for different (dataset, task, train_mode).
    """
    if args.batch_size is None:
        if args.stage == "pretrain":
            args.batch_size = 256
        else:
            args.batch_size = 128

    return args


def set_auto_params(args):
    """Automatically set the parameters for the experiment."""
    # gpu configuration
    if args.gpu is None:
        args.gpu = 0
    args.device = select_device(str(args.gpu))
    args.half = False  # half precision only supported on CUDA

    # retrieve the user name
    args.username = get_username()

    # set downstream task
    args.task = set_task(args)

    # parse the model yaml file
    dataset_yaml = f"./data/{args.dataset}.yaml"
    args.dataset_config = load_yaml(dataset_yaml)
    
    args.sequence_sampler = True if args.learn_framework in {"FOCAL"} else False

    # dataloader config
    args.workers = 10

    # set the train mode
    args.train_mode = get_train_mode(args.learn_framework)

    # set batch size
    args = set_batch_size(args)

    # set output path
    args = set_model_weight_folder(args)
    args = set_model_weight_file(args)
    args = set_output_paths(args)

    return args
