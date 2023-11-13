import os
import torch
import logging
import numpy as np
from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging
from train_utils.optimizer import define_optimizer
from train_utils.lr_scheduler import define_lr_scheduler

# utils
from general_utils.time_utils import time_sync
from general_utils.weight_utils import load_model_weight, set_learnable_params_finetune
from params.output_paths import set_finetune_weights


def finetune(
    args,
    classifier,
    augmenter,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    classifier_loss_func,
    num_batches,
):
    """Fine tune the backbone network with only the class layer."""
    # Load the pretrained feature extractor
    pretrain_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_pretrain_latest.pt")
    classifier = load_model_weight(args, classifier, pretrain_weight, load_class_layer=False)
    learnable_parameters = set_learnable_params_finetune(args, classifier)

    # Init the optimizer, scheduler, and weight files
    optimizer = define_optimizer(args, learnable_parameters)
    lr_scheduler = define_lr_scheduler(args, optimizer)
    best_weight, latest_weight = set_finetune_weights(args)

    # Training loop
    logging.info("---------------------------Start Fine Tuning-------------------------------")
    start = time_sync()
    best_val_acc = 0

    val_epochs = 5
    for epoch in range(args.dataset_config[args.learn_framework]["finetune_lr_scheduler"]["train_epochs"]):
        if epoch > 0:
            logging.info("-" * 40 + f"Epoch {epoch}" + "-" * 40)

        # set model to train mode
        classifier.train()

        # training loop
        train_loss_list = []
        for i, (time_loc_inputs, labels) in tqdm(enumerate(train_dataloader), total=num_batches):
            # move to target device, FFT, and augmentations
            aug_freq_loc_inputs, labels = augmenter.forward("no", time_loc_inputs, labels)

            # forward pass
            logits = classifier(aug_freq_loc_inputs)
            loss = classifier_loss_func(logits, labels)

            # back propagation
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss_list.append(loss.item())

        # validation and logging
        if epoch % val_epochs == 0:
            train_loss = np.mean(train_loss_list)
            val_metric, val_loss = val_and_logging(
                args,
                epoch,
                classifier,
                augmenter,
                val_dataloader,
                test_dataloader,
                classifier_loss_func,
                train_loss,
            )

            # Save the latest model
            torch.save(classifier.state_dict(), latest_weight)

            # Save the best model according to validation result
            if val_metric > best_val_acc:
                best_val_acc = val_metric
                torch.save(classifier.state_dict(), best_weight)

        # Update the learning rate scheduler
        lr_scheduler.step(epoch)

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
