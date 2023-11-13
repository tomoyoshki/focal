import torch.nn as nn

# FOCAL Backbone models
from models.DeepSense import DeepSense
from models.SW_Transformer import SW_Transformer

# FOCAL Frameworks
from models.FOCALModules import FOCAL

# FOCAL Loss
from models.loss import FOCALLoss


def init_backbone_model(args):
    """Automatically select the model according to args."""
    if args.model == "DeepSense":
        classifier = DeepSense(args)
    elif args.model == "SW_Transformer":
        classifier = SW_Transformer(args)
    else:
        raise Exception(f"Invalid model provided: {args.model}")

    classifier = classifier.to(args.device)

    return classifier


def init_contrastive_framework(args, backbone_model):
    if args.learn_framework == "FOCAL":
        default_model = FOCAL(args, backbone_model)
    else:
        raise NotImplementedError(f"Invalid {args.train_mode} framework {args.learn_framework} provided")

    default_model = default_model.to(args.device)

    return default_model

def init_pretrain_framework(args, backbone_model):
    """
    Initialize the pretraining framework according to args.
    """
    default_model = init_contrastive_framework(args, backbone_model)

    return default_model


def init_loss_func(args):
    """Initialize the loss function according to the config."""
    if args.train_mode == "supervised" or args.stage == "finetune":
        loss_func = nn.CrossEntropyLoss()
    elif args.train_mode == "contrastive":
        if args.learn_framework in {"FOCAL"}:
            loss_func = FOCALLoss(args).to(args.device)
        else:
            raise NotImplementedError(f"Invalid {args.train_mode} framework {args.learn_framework} provided")
    else:
        raise Exception(f"Invalid train mode provided: {args.train_mode}")

    return loss_func
