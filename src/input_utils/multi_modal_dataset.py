import os
import torch
import numpy as np

from torch.utils.data import Dataset
from random import shuffle


class MultiModalDataset(Dataset):
    def __init__(self, args, index_file, label_ratio=1):
        """
        Args:
            modalities (_type_): The list of modalities
            classes (_type_): The list of classes
            index_file (_type_): The list of sample file names
            sample_path (_type_): The base sample path.

        Sample:
            - label: Tensor
            - flag
                - phone
                    - audio: True
                    - acc: False
            - data:
                -phone
                    - audio: Tensor
                    - acc: Tensor
        """
        self.args = args
        self.sample_files = list(np.loadtxt(index_file, dtype=str))

        if label_ratio < 1:
            shuffle(self.sample_files)
            self.sample_files = self.sample_files[: round(len(self.sample_files) * label_ratio)]

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample = torch.load(self.sample_files[idx])
        data = sample["data"]

        # ACIDS and Parkland
        if isinstance(sample["label"], dict):
            if self.args.task == "vehicle_classification":
                label = sample["label"]["vehicle_type"]
            elif self.args.task == "distance_classification":
                label = sample["label"]["distance"]
            elif self.args.task == "speed_classification":
                label = sample["label"]["speed"]
            else:
                raise ValueError(f"Unknown task: {self.args.task}")
        else:
            label = sample["label"]

        return data, label

class MultiModalSequenceDataset(Dataset):
    def __init__(self, args, index_file):
        """
        Extract multiple sequences of consecutive samples at the time dimension.
        """
        self.args = args
        self.sample_files = list(np.loadtxt(index_file, dtype=str))
        self.partition_subsequences()

    def partition_subsequences(self):
        """
        Extract all sequence IDs from the sample files.
        seq_to_sample: {sequence_id: [(sample_id, sample_file), ...], ...}
        """
        seq_len = self.args.dataset_config["seq_len"]

        if self.args.dataset == "RealWorld_HAR":
            delimiter = "-"
        else:
            delimiter = "_"

        seq_to_samples = {}
        for sample_idx, sample_file in enumerate(self.sample_files):
            # Sequence ID is separeted by the last underscore symbol.
            basename = os.path.basename(sample_file)
            seq = basename.rsplit(delimiter, 1)[0]

            if seq not in seq_to_samples:
                seq_to_samples[seq] = [(sample_idx, sample_file)]
            else:
                seq_to_samples[seq].append((sample_idx, sample_file))

        # sort the sequences
        for seq in seq_to_samples:
            seq_to_samples[seq].sort(key=lambda x: int(os.path.basename(x[1]).rsplit(delimiter, 1)[1].split(".")[0]))
            seq_to_samples[seq] = [e[0] for e in seq_to_samples[seq]]

        # divide sequences into subsequences of fixed length
        self.subseqs = []
        self.subseq_to_sample_idx = {}
        for seq in seq_to_samples:
            for i in range(0, len(seq_to_samples[seq]), seq_len):
                subseq = f"{seq}_{i}"
                self.subseqs.append(subseq)

                # constitute the sample list with fixed length
                sample_id_list = seq_to_samples[seq][i : i + seq_len]
                while len(sample_id_list) < seq_len:
                    sample_id_list.append(sample_id_list[-1])

                self.subseq_to_sample_idx[subseq] = sample_id_list

    def __len__(self):
        return len(self.subseqs)

    def __getitem__(self, sample_idx):
        """
        Extract a random sequence of samples.
        """
        sample = torch.load(self.sample_files[sample_idx])
        data = sample["data"]

        if isinstance(sample["label"], dict):
            if self.args.task == "vehicle_classification":
                label = sample["label"]["vehicle_type"]
            elif self.args.task == "distance_classification":
                label = sample["label"]["distance"]
            elif self.args.task == "speed_classification":
                label = sample["label"]["speed"]
            else:
                raise ValueError(f"Unknown task: {self.args.task}")
        else:
            label = sample["label"]

        return data, label