import os
import torch
from preprocessing_config import output_directory

from tqdm import tqdm


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


def partition_data(supervised_train_index, paired_data_path, output_path):
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

    for sample in tqdm(data_samples):
        file_path = os.path.join(os.path.join(paired_data_path, sample))

        """For all users, we only preserve samples with complete modalities in the dataset."""
        load_sample = torch.load(file_path)
        complete_modality_flag = 1
        for loc in load_sample["flag"]:
            for mod in load_sample["flag"][loc]:
                complete_modality_flag *= load_sample["flag"][loc][mod]

        if complete_modality_flag:
            train_samples.append(file_path)

    # save the index file
    print(f"Number of extra training samples: {len(train_samples)}")
    with open(os.path.join(output_path, "pretrain_index.txt"), "w") as f:
        for sample_file in train_samples:
            f.write(sample_file + "\n")

        with open(supervised_train_index, "r") as fs:
            for line in fs:
                f.write(line)

if __name__ == "__main__":
    paired_data_path = os.path.join(output_directory, "extra_time_samples")
    output_path = os.path.join(output_directory, "time_data_partition")
    supervised_train_index = os.path.join(output_path, "train_index.txt")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # partition the dta
    partition_data(supervised_train_index, paired_data_path, output_path)
