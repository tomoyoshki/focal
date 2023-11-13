import torch
import logging
import numpy as np
from tqdm import tqdm

# utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from train_utils.knn import extract_sample_features
from train_utils.loss_calc_utils import calc_pretrain_loss   

def eval_task_metrics(args, labels, predictions):
    """Evaluate the downstream task metrics."""
    if args.task in {"distance_classification", "speed_classification"}:
        num_classes = args.dataset_config[args.task]["num_classes"]
        mean_acc = 1 - (np.abs(labels-predictions) / np.maximum(labels, (num_classes - 1) - labels))
        mean_acc = np.nan_to_num(mean_acc, nan=1.0)
        mean_acc = mean_acc.mean()
    else:    
        mean_acc = accuracy_score(labels, predictions)
    mean_f1 = f1_score(labels, predictions, average="macro", zero_division=1)
    try:
        conf_matrix = confusion_matrix(labels, predictions)
    except:
        conf_matrix = []

    return mean_acc, mean_f1, conf_matrix


def eval_supervised_model(args, classifier, augmenter, dataloader, loss_func):
    classifier.eval()
    # iterate over all batches
    num_batches = len(dataloader)
    classifier_loss_list = []
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for i, (time_loc_inputs, labels) in tqdm(enumerate(dataloader), total=num_batches):
            # move to target device, FFT, and augmentations
            freq_loc_inputs, labels = augmenter.forward("no", time_loc_inputs, labels)

            # forward pass
            logits = classifier(freq_loc_inputs)
            classifier_loss_list.append(loss_func(logits, labels).item())

            predictions = logits.argmax(dim=1, keepdim=False)
            labels = labels.argmax(dim=1, keepdim=False) if labels.dim() > 1 else labels

            # for future computation of acc or F1 score
            saved_predictions = predictions.cpu().numpy()
            saved_labels = labels.cpu().numpy()
            all_predictions.append(saved_predictions)
            all_labels.append(saved_labels)

    # calculate mean loss
    mean_classifier_loss = np.mean(classifier_loss_list)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # calculate the classification metrics
    metrics = eval_task_metrics(args, all_labels, all_predictions)

    return mean_classifier_loss, metrics


def eval_pretrained_model(args, default_model, estimator, augmenter, dataloader, loss_func):
    """Evaluate the downstream task performance with KNN estimator."""
    default_model.eval()

    sample_embeddings = []
    labels = []
    loss_list = []
    with torch.no_grad():
        for time_loc_inputs, label in tqdm(dataloader, total=len(dataloader)):
            """Move idx to target device, save label"""
            label = label.argmax(dim=1, keepdim=False) if label.dim() > 1 else label
            labels.append(label.cpu().numpy())

            """Eval pretrain loss."""
            loss = calc_pretrain_loss(args, default_model, augmenter, loss_func, time_loc_inputs).item()
            loss_list.append(loss)

            """Eval KNN estimator."""
            aug_freq_loc_inputs = augmenter.forward("no", time_loc_inputs)
            feat = extract_sample_features(args, default_model.backbone, aug_freq_loc_inputs)
            sample_embeddings.append(feat.detach().cpu().numpy())

    # knn predictions
    sample_embeddings = np.concatenate(sample_embeddings)
    labels = np.concatenate(labels)
    predictions = torch.Tensor(estimator.predict(sample_embeddings))
    predictions = predictions.argmax(dim=1, keepdim=False) if predictions.dim() > 1 else predictions

    # compute metrics
    mean_loss = np.mean(loss_list)
    metrics = eval_task_metrics(args, labels, predictions)

    return mean_loss, metrics

def val_and_logging(
    args,
    epoch,
    model,
    augmenter,
    val_loader,
    test_loader,
    loss_func,
    train_loss,
    estimator=None,
):
    if args.train_mode in {"contrastive"} and args.stage == "pretrain":
        logging.info(f"Train {args.train_mode} loss: {train_loss: .5f} \n")
    else:
        logging.info(f"Training loss: {train_loss: .5f} \n")

    if args.train_mode == "supervised" or args.stage == "finetune":
        """Supervised training or fine-tuning"""
        val_loss, val_metrics = eval_supervised_model(args, model, augmenter, val_loader, loss_func)
        test_loss, test_metrics = eval_supervised_model(args, model, augmenter, test_loader, loss_func)
    else:
        val_loss, val_metrics = eval_pretrained_model(args, model, estimator, augmenter, val_loader, loss_func)
        test_loss, test_metrics = eval_pretrained_model(args, model, estimator, augmenter, test_loader, loss_func)


    logging.info(f"Val loss: {val_loss: .5f}")
    logging.info(f"Val acc: {val_metrics[0]: .5f}, val f1: {val_metrics[1]: .5f}")
    logging.info(f"Val confusion matrix:\n {val_metrics[2]} \n")
    logging.info(f"Test loss: {test_loss: .5f}")
    logging.info(f"Test acc: {test_metrics[0]: .5f}, test f1: {test_metrics[1]: .5f}")
    logging.info(f"Test confusion matrix:\n {test_metrics[2]} \n")

    return val_metrics[0], val_loss
