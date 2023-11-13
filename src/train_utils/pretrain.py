import os
import torch
import logging
import numpy as np

from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging
from train_utils.optimizer import define_optimizer
from train_utils.lr_scheduler import define_lr_scheduler
from train_utils.knn import compute_knn
from train_utils.model_selection import init_pretrain_framework
from train_utils.loss_calc_utils import calc_pretrain_loss
from general_utils.weight_utils import freeze_patch_embedding

# utils
from general_utils.time_utils import time_sync


def pretrain(
    args,
    backbone_model,
    augmenter,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    loss_func,
    num_batches,
):
    """
    The supervised training function for tbe backbone network,
    used in train of supervised mode or fine-tune of foundation models.
    """
    # Initialize contrastive model
    default_model = init_pretrain_framework(args, backbone_model)

    # Define the optimizer and learning rate scheduler
    optimizer = define_optimizer(args, default_model.parameters())
    lr_scheduler = define_lr_scheduler(args, optimizer)

    default_model = freeze_patch_embedding(args, default_model)

    # Training loop
    logging.info("---------------------------Start Pretraining Classifier-------------------------------")
    start = time_sync()
    best_val_loss = np.inf
    best_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_best.pt")
    latest_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_latest.pt")

    for epoch in range(args.dataset_config[args.learn_framework]["pretrain_lr_scheduler"]["train_epochs"]):
        if epoch > 0:
            logging.info("-" * 40 + f"Epoch {epoch}" + "-" * 40)

        # set model to train mode
        default_model.train()

        # training loop
        train_loss_list = []

        # regularization configuration
        for i, (time_loc_inputs, _) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # clear the gradients
            optimizer.zero_grad()

            # move to target device, FFT, and augmentations
            loss = calc_pretrain_loss(args, default_model, augmenter, loss_func, time_loc_inputs)

            # back propagation
            loss.backward()

            # update
            optimizer.step()
            train_loss_list.append(loss.item())

        if epoch % 10 == 0:
            # Use KNN classifier for validation
            knn_estimator = compute_knn(args, default_model.backbone, augmenter, train_dataloader)

            # validation and logging
            train_loss = np.mean(train_loss_list)
            val_acc, val_loss = val_and_logging(
                args,
                epoch,
                default_model,
                augmenter,
                val_dataloader,
                test_dataloader,
                loss_func,
                train_loss,
                estimator=knn_estimator,
            )

            # Save the latest model, only the backbone parameters are saved
            torch.save(default_model.backbone.state_dict(), latest_weight)

            # Save the best model according to validation result
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(default_model.backbone.state_dict(), best_weight)

        # Update the learning rate scheduler
        lr_scheduler.step(epoch)

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
