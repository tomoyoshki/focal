import warnings

warnings.simplefilter("ignore", UserWarning)

import logging
import torch
import numpy as np

import sys

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

# train utils
from train_utils.supervised_train import supervised_train
from train_utils.pretrain import pretrain
from train_utils.finetune import finetune

# utils
from params.train_params import parse_train_params
from input_utils.multi_modal_dataloader import create_dataloader
from train_utils.model_selection import init_backbone_model, init_loss_func


def train(args):
    """The specific function for training."""
    # Init data loaders
    train_dataloader = create_dataloader("train", args, batch_size=args.batch_size, workers=args.workers)
    val_dataloader = create_dataloader("val", args, batch_size=args.batch_size, workers=args.workers)
    test_dataloader = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)
    num_batches = len(train_dataloader)

    logging.info(f"{'='*30}Dataloaders loaded{'='*30}")

    # Only import augmenter after parse arguments so the device is correct
    from data_augmenter.Augmenter import Augmenter

    # Init the miss modality simulator
    augmenter = Augmenter(args)
    augmenter.to(args.device)
    args.augmenter = augmenter

    # Init the classifier model
    classifier = init_backbone_model(args)
    args.classifier = classifier

    # define the loss function
    loss_func = init_loss_func(args)

    if args.train_mode == "supervised":
        supervised_train(
            args,
            classifier,
            augmenter,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            loss_func,
            num_batches,
        )
    elif args.stage == "pretrain":
        pretrain(
            args,
            classifier,
            augmenter,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            loss_func,
            num_batches,
        )
    elif args.stage == "finetune":
        finetune(
            args,
            classifier,
            augmenter,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            loss_func,
            num_batches,
        )
    else:
        raise Exception("Invalid stage ({args.stage}) provided.")


def main_train():
    """The main function of training"""
    args = parse_train_params()
    train(args)


if __name__ == "__main__":
    main_train()
