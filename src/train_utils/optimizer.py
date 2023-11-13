from torch import optim as optim

def define_optimizer(args, parameters):
    """Define the optimizer."""
    if args.train_mode in {"supervised"}:
        optimizer_config = args.dataset_config[args.model]["optimizer"]
    elif args.stage == "pretrain":
        optimizer_config = args.dataset_config[args.learn_framework]["pretrain_optimizer"]
    elif args.stage == "finetune":
        optimizer_config = args.dataset_config[args.learn_framework]["finetune_optimizer"]
    else:
        raise Exception("Optimizer not defined.")
    optimizer_name = optimizer_config["name"]
    
    # select weight decay
    if isinstance(optimizer_config["weight_decay"], dict):
        weight_decay = optimizer_config["weight_decay"][args.model]
    else:
        weight_decay = optimizer_config["weight_decay"]

    if optimizer_name == "Adam":
        optimizer = optim.Adam(
            parameters,
            lr=optimizer_config["start_lr"],
            weight_decay=weight_decay,
        )
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            parameters,
            lr=optimizer_config["start_lr"],
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} not implemented.")

    return optimizer