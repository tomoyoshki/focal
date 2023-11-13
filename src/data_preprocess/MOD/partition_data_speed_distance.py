from cgi import test
import os
import torch
import random
import getpass

import numpy as np

from tqdm import tqdm
from extract_samples_speed_distance import parse_labels
from preprocessing_config import output_directory, distance_speed_option as option


def extract_user_list(input_path):
    """Extract the user list in the given path.

    Args:
        input_path (_type_): _description_
    """
    user_list = []

    for e in os.listdir(input_path):
        if os.path.isdir(os.path.join(input_path, e)):
            user_list.append(e)

    return user_list


def partition_data(option, paired_data_path, output_path, train_ratio=0.9):
    """Partition the data according to the given ratio, using all-but-one strategy.
    We don't touch the processed data, but only save a new index file.

    Args:
        paired_data_path (_type_): _description_
        output_path (_type_): _description_
        train_ratio (_type_): _description_
    """
    # for users in training set, only preserve their data samples with complete modalities
    data_samples = os.listdir(paired_data_path)
    train_samples = []
    val_samples = []
    test_samples = []

    for sample in tqdm(data_samples):
        file_path = os.path.join(os.path.join(paired_data_path, sample))
        label = parse_labels(os.path.basename(file_path))

        # only preserve mustang
        if label["vehicle"] != 5:
            continue

        if random.random() < train_ratio:
            target = train_samples
        else:
            target = test_samples

        """For all users, we only preserve samples with complete modalities in the dataset."""
        load_sample = torch.load(file_path)
        complete_modality_flag = 1
        for loc in load_sample["flag"]:
            for mod in load_sample["flag"][loc]:
                complete_modality_flag *= load_sample["flag"][loc][mod]

        if complete_modality_flag:
            target.append(file_path)

    # same val as test examples
    val_samples = test_samples

    # save the index file
    print(
        f"Number of training samples: {len(train_samples)}, \
        number of validation samples: {len(val_samples)}, \
        number of testing samples: {len(test_samples)}."
    )
    with open(os.path.join(output_path, "train_index.txt"), "w") as f:
        for sample_file in train_samples:
            f.write(sample_file + "\n")
    with open(os.path.join(output_path, "val_index.txt"), "w") as f:
        for sample_file in val_samples:
            f.write(sample_file + "\n")
    with open(os.path.join(output_path, "test_index.txt"), "w") as f:
        for sample_file in test_samples:
            f.write(sample_file + "\n")

if __name__ == "__main__":
    paired_data_path = os.path.join(output_directory, "distance_speed_individual_time_samples")

    if option == "distance_classification":
        output_path = os.path.join(output_directory, "distance_data_partition")
    elif option == "speed_classification":
        output_path = os.path.join(output_directory, "speed_data_partition")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # partition the dta
    partition_data(option, paired_data_path, output_path)
