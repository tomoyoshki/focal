import os
import warnings

warnings.simplefilter("ignore", UserWarning)

import torch.nn as nn

# utils
from general_utils.weight_utils import load_model_weight
from params.test_params import parse_test_params
from input_utils.multi_modal_dataloader import create_dataloader
from train_utils.eval_functions import eval_supervised_model
from train_utils.model_selection import init_backbone_model


def test(args):
    """The main function for test."""
    # Init data loaders
    test_dataloader = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)

    # Only import augmenter after parse arguments so the device is correct
    from data_augmenter.Augmenter import Augmenter

    # Init the miss modality simulator
    augmenter = Augmenter(args)
    augmenter.to(args.device)
    args.augmenter = augmenter

    # Init the classifier model
    classifier = init_backbone_model(args)
    classifier = load_model_weight(args, classifier, args.classifier_weight, load_class_layer=True)
    args.classifier = classifier
    # define the loss function
    classifier_loss_func = nn.CrossEntropyLoss()

    test_classifier_loss, test_metrics = eval_supervised_model(
        args, classifier, augmenter, test_dataloader, classifier_loss_func
    )
    if "regression" in args.task:
        print(f"Test classifier loss: {test_classifier_loss: .5f}, test mse: {test_metrics[0]: .5f}")
        return test_classifier_loss, test_metrics[0]
    else:
        print(f"Test classifier loss: {test_classifier_loss: .5f}")
        print(f"Test acc: {test_metrics[0]: .5f}, test f1: {test_metrics[1]: .5f}")
        print(f"Test confusion matrix:\n {test_metrics[2]}")

        return test_classifier_loss, test_metrics[0], test_metrics[1]


def main_test():
    """The main function of training"""
    args = parse_test_params()

    test(args)


if __name__ == "__main__":
    main_test()
