import os
import torch
import logging

from params.output_paths import find_most_recent_weight
from params.params_util import get_train_mode


def load_model_weight(args, model, weight_file, load_class_layer=True):
    """Load the trained model weight into the model.

    Args:
        model (_type_): _description_
        weight_file (_type_): _description_
    """
    trained_dict = torch.load(weight_file, map_location=args.device)
    model_dict = model.state_dict()
    if load_class_layer:
        load_dict = {k: v for k, v in trained_dict.items() if k in model_dict}
    else:
        load_dict = {k: v for k, v in trained_dict.items() if k in model_dict and "class_layer" not in k}
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)

    return model


def load_feature_extraction_weight(args, model):
    """
    Only load the feature extraction parameters for fusion-based pretraining.
    NOTE: The feature extract weight is assumed to be pretrained with a contrastive framework.
    """
    config = args.dataset_config[args.learn_framework]

    # find pretrained weight for feature extraction part
    feature_learn_framework = config["feature_learn_framework"]
    newest_id, newest_weight_path = find_most_recent_weight(
        args.debug, args.dataset, args.model, get_train_mode(feature_learn_framework), feature_learn_framework
    )
    if newest_id < 0:
        raise Exception(f"No pretrained weight for feature extraction part found for {feature_learn_framework}.")
    logging.info(f"=\tLoading pretrained weight for feature extraction from: {newest_weight_path}")
    most_recent_weight = os.path.join(newest_weight_path, f"{args.dataset}_{args.model}_pretrain_best.pt")
    trained_dict = torch.load(most_recent_weight)

    # find the modality specific layers
    load_dict = {}
    modalities = args.dataset_config["modality_names"]
    model_dict = model.backbone.state_dict()
    for k, v in trained_dict.items():
        if k in model_dict:
            for mod in modalities:
                if mod in k:
                    load_dict[k] = v

    # load the feature extraction part
    model_dict.update(load_dict)
    model.backbone.load_state_dict(model_dict)

    return model


def set_learnable_params_finetune(args, classifier):
    """
    Set the learnable parameters for fine-tuning.
    """
    learnable_parameters = []
    for name, param in classifier.named_parameters():
        if args.learn_framework in {"FOCAL"}:
            if "class_layer" in name or "mod_fusion_layer" in name:
                param.requires_grad = True
                learnable_parameters.append(param)
            else:
                param.requires_grad = False
        else:
            if "class_layer" in name:
                param.requires_grad = True
                learnable_parameters.append(param)
            else:
                param.requires_grad = False

    return learnable_parameters


def freeze_patch_embedding(args, default_model):
    """
    Freeze the patch embedding layaer.
    """
    if "Fusion" not in args.learn_framework:
        for name, param in default_model.backbone.named_parameters():
            if "patch_embed" in name:
                param.requires_grad = False

    return default_model
