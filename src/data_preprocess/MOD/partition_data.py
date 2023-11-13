import os
import torch
import random


from tqdm import tqdm

from preprocessing_config import output_directory

PRESERVED_FOLDERS = {
    # "bicycle2",
    "motor",
    "mustang0528",
    "walk",
    "walk2",
    # "pickup",
    "tesla",
    "Polaris0150pm",
    "Polaris0215pm",
    "Polaris0235pm-NoLineOfSight",
    "Warhog1135am",
    "Warhog1149am",
    "Warhog-NoLineOfSight",
    "Silverado0255pm",
    "Silverado0315pm",
}


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


def partition_data(paired_data_path, output_path, shakes, train_ratio=0.8):
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
        sample_shake = os.path.basename(sample).split("_")[1]
        if sample_shake not in shakes:
            continue

        sample_target = os.path.basename(sample).split("_")[0]
        if sample_target not in PRESERVED_FOLDERS:
            continue

        file_path = os.path.join(os.path.join(paired_data_path, sample))

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
    paired_data_path = os.path.join(output_directory, "individual_time_samples")
    output_path = os.path.join(output_directory, "time_data_partition")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # extract user list
    shakes = ["rs1", "rs2", "rs3", "rs7"]
    # targets = []

    # partition the dta
    partition_data(paired_data_path, output_path, shakes)
