import os
import json
import logging


def check_paths(path_list):
    """
    Check the given path list.
    :param path_list:
    :return:
    """
    for p in path_list:
        if not os.path.exists(p):
            os.mkdir(p)


def remove_files(path_list):
    """
    Remove the given path list.
    :param path_list:
    :return:
    """
    for p in path_list:
        if os.path.exists(p):
            os.remove(p)


def set_model_weight_suffix(
    train_mode,
    learn_framework=None,
    task=None,
    label_ratio=None,
    miss_modalities=None,
    tag=None,
):
    """Automatically get the model path.

    Args:
        args (_type_): _description_
    """
    if train_mode == "supervised":
        suffix = f"supervised_{task}_{label_ratio}"
    elif train_mode in {"contrastive"}:
        suffix = f"{train_mode}_{learn_framework}"
    else:
        raise Exception(f"Unknown train mode: {train_mode}")

    if tag is not None:
        suffix += f"-{tag}"

    return suffix


def find_most_recent_weight(
    dataset,
    model,
    train_mode,
    learn_framework,
    task=None,
    label_ratio=None,
    return_suffix=False,
    tag=None,
):
    """Find the most recent weight path for the given (model, train_mode, framework).)"""
    # base model path
    base_path = f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/weights"
    dataset_model_path = os.path.join(base_path, f"{dataset}_{model}")

    # identify the proper suffix
    suffix = set_model_weight_suffix(train_mode, learn_framework, task, label_ratio, tag=tag)

    # find the most recent weight (training, finetuning, testing)
    newest_id = -1
    newest_weight = None
    existing_weights = os.listdir(dataset_model_path)
    for weight in existing_weights:
        # only check qualified weight with the required suffix
        weight_suffix = weight.split("_", 1)[-1]
        if suffix != weight_suffix:
            continue
        else:
            weight_id = int(weight.split("_")[0][3:])
            if weight_id > newest_id:
                newest_id = weight_id
                newest_weight = os.path.join(dataset_model_path, weight)

    if return_suffix:
        return newest_id, newest_weight, suffix
    else:
        return newest_id, newest_weight


def set_model_weight_folder(args):
    """Automatically get the model path.

    Args:
        args (_type_): _description_
    """
    base_path = f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/weights"
    dataset_model_path = os.path.join(base_path, f"{args.dataset}_{args.model}")
    check_paths([base_path, dataset_model_path])

    # get the newest id matching the current config
    newest_id, newest_weight, suffix = find_most_recent_weight(
        args.dataset,
        args.model,
        args.train_mode,
        args.learn_framework,
        args.task,
        args.label_ratio,
        return_suffix=True,
        tag=args.tag,
    )

    # set the weight path to avoid redundancy
    if args.option == "test" or args.stage == "finetune":
        if args.model_weight is not None:
            weight_folder = args.model_weight
        else:
            if newest_id == -1:
                raise Exception(f"No existing model weights for {suffix}")
            else:
                weight_folder = newest_weight
    else:
        weight_folder = os.path.join(dataset_model_path, f"exp{newest_id + 1}") + f"_{suffix}"
        check_paths([weight_folder])
        model_config = args.dataset_config[args.model]
        with open(os.path.join(weight_folder, "model_config.json"), "w") as f:
            f.write(json.dumps(model_config, indent=4))

        if args.train_mode != "supervised" and args.stage == "pretrain":
            framework_config_log = os.path.join(weight_folder, "learn_framework_config.json")
            with open(framework_config_log, "w") as f:
                f.write(json.dumps(args.dataset_config[args.learn_framework], indent=4))

    # set log files
    if args.option == "train":
        if args.train_mode == "supervised":
            args.train_log_file = os.path.join(weight_folder, f"train_log.txt")
            args.tensorboard_log = os.path.join(weight_folder, f"train_events")
        else:
            if args.stage == "pretrain":
                args.train_log_file = os.path.join(weight_folder, f"pretrain_log.txt")
                args.tensorboard_log = os.path.join(weight_folder, f"pretrain_events")
            else:
                args.train_log_file = os.path.join(
                    weight_folder, f"{args.task}_{args.label_ratio}_{args.stage}_log.txt"
                )
                args.tensorboard_log = os.path.join(
                    weight_folder, f"{args.task}_{args.label_ratio}_{args.stage}_events"
                )
        # delete old log file
        remove_files([args.train_log_file])
        # set logging config
        logging.basicConfig(
            level=logging.INFO, handlers=[logging.FileHandler(args.train_log_file), logging.StreamHandler()], force=True
        )
        logging.info(f"=\t[Model weights path]: {weight_folder}")

    args.weight_folder = weight_folder

    return args


def set_model_weight_file(args):
    """Automatically select the classifier weight during the testing"""
    if args.train_mode == "supervised":
        args.classifier_weight = os.path.join(
            args.weight_folder,
            f"{args.dataset}_{args.model}_{args.task}_best.pt",
        )
    elif args.train_mode in {"contrastive"}:
        if args.stage == "pretrain":
            args.classifier_weight = os.path.join(
                args.weight_folder,
                f"{args.dataset}_{args.model}_pretrain_best.pt",
            )
        else:
            args.classifier_weight = os.path.join(
                args.weight_folder,
                f"{args.dataset}_{args.model}_{args.task}_{args.label_ratio}_finetune_best.pt",
            )
    else:
        raise Exception(f"Invalid training mode provided: {args.stage}")

    return args


def set_finetune_weights(args):
    """Automatically select the finetune weight during the testing"""
    best_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.model}_{args.task}_{args.label_ratio}_finetune_best.pt",
    )
    latest_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.model}_{args.task}_{args.label_ratio}_finetune_latest.pt",
    )
    return best_weight, latest_weight


def set_output_paths(args):
    result_root_path = f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/result"
    log_root_path = os.path.join(result_root_path, "log")
    args.log_path = os.path.join(log_root_path, f"{args.dataset}_{args.model}_{args.train_mode}")
    check_paths([result_root_path, log_root_path, args.log_path])
    return args
