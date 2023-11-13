import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def extract_sample_features(args, classifier, aug_freq_loc_inputs, proj_head=False):
    """
    Compute the sample features for the given input.
    proj_head: whether to extract features after the projection head.
    """
    if args.learn_framework in {"FOCAL"}:
        mod_features = classifier(aug_freq_loc_inputs, class_head=False, proj_head=proj_head)
        mod_features = [mod_features[mod] for mod in args.dataset_config["modality_names"]]
        features = torch.cat(mod_features, dim=1)
    else:
        raise Exception(f"Invalid learn framework ({args.learn_framework}) provided")

    return features


def compute_knn(args, classifier, augmenter, data_loader_train):
    classifier.eval()

    sample_embeddings = []
    labels = []
    for time_loc_inputs, y in data_loader_train:
        # FFT and move to target device
        aug_freq_loc_inputs, _ = augmenter.forward("no", time_loc_inputs, y)

        # feature extraction
        features = extract_sample_features(args, classifier, aug_freq_loc_inputs)
        sample_embeddings.append(features.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    sample_embeddings = np.concatenate(sample_embeddings)
    labels = np.concatenate(labels)

    estimator = KNeighborsClassifier()
    estimator.fit(sample_embeddings, labels)

    return estimator
