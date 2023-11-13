import os
import time
import torch
import numpy as np

import torchaudio.transforms as T
from scipy.io import loadmat

from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import cpu_count
from preprocessing_config import output_directory, distance_speed_input_path as input_path

"""
Configuration:
    1) Sampling frequency:
        - acoustic: 1025.641 Hz
        - acc, seismic: 1025.641 Hz
    4) We save the aligned time-series for each sensor in shape [channel, interval, spectrum (time series)].
"""

SEGMENT_SPAN = 2
INTERVAL_SPAN = 0.2
SEGMENT_OVERLAP_RATIO = 0.0
INTERVAL_OVERLAP_RATIO = 0.0
STD_THRESHOLD = 0
AUD_DOWNSAMPLE_RATE = 2

FREQS = {"audio": 16000 / AUD_DOWNSAMPLE_RATE, "seismic": 100, "acc": 100}
VEHICLE_TYPE_LABELS = {
    "Polaris": 0,
    "Warhog": 1,
    "Silverado": 2,
    "motor": 3,
    "tesla": 4,
    "mustang": 5,
    "walk": 6,
    "bicycle": 7,
    "forester": 8,
    "pickup": 9,
    "scooter": 10,
}

SPEED_LABELS = {
    "5mph": 0,
    "10mph": 1,
    "15mph": 2,
}

DISTANCE_LABELS = {
    "distance1": 0,
    "distance2": 1,
    "distance3": 2,
    "distance15": 0,
    "distance37.5": 1,
    "distance67.5": 2,
}


def parse_labels(input_folder):
    """Parse the input folder list"""
    label = {}

    # parse vehicle type
    vehicle_type = input_folder.split("_")[0]
    label["vehicle"] = VEHICLE_TYPE_LABELS[vehicle_type]

    # parse speed
    tokens = input_folder.split("_")
    for token in tokens:
        if "mph" in token:
            label["speed"] = SPEED_LABELS[token]
            break

    # parse distance
    for token in tokens:
        if "distance" in token:
            label["distance"] = DISTANCE_LABELS[token]
            break

    return label


def resample_numpy_array(raw_audio, orig_freq, new_freq):
    """Resample the given audio array with torchaudio APIs.

    Args:
        raw_audio (_type_): [time, channel]
    """
    # input, [time, channel] --> [channel, time]
    audio_tensor = torch.from_numpy(raw_audio)
    audio_tensor = torch.transpose(audio_tensor, 0, 1)

    # transform
    resampler = T.Resample(orig_freq, new_freq, dtype=float)
    new_audio_tensor = resampler(audio_tensor)

    # output, [channel, time] --> [time, channel]
    new_audio_tensor = torch.transpose(new_audio_tensor, 0, 1)
    new_audio_array = new_audio_tensor.numpy()

    return new_audio_array


def split_array_with_overlap(input, overlap_ratio, interval_len=None, num_interval=None):
    """Split the input array into num intervals with overlap ratio.

    Args:
        input (_type_): [freq * sec, channel]
        num_interval (_type_): 39
        overlap_ratio (_type_): 0.5
    """
    assert (interval_len is not None) or (num_interval is not None)

    if interval_len is None:
        interval_len = int(len(input) // (1 + (num_interval - 1) * (1 - overlap_ratio)))
    else:
        interval_len = int(interval_len)

    splitted_input = []
    for start in range(0, int(len(input) - interval_len + 1), int((1 - overlap_ratio) * interval_len)):
        interval = input[start : start + interval_len]

        # only prserve data wth complete length
        if len(interval) == interval_len:
            splitted_input.append(input[start : start + interval_len])
    splitted_input = np.array(splitted_input)

    return splitted_input


def extract_loc_mod_tensor(raw_data, segment_len, freq):
    """Extract the Tensor for a given location and sensor.
    We assume the data is interpolated before. No time dimension is included.

    Args:
        raw_data (_type_): _description_
        loc (_type_): _description_
        modality (_type_): _description_
    """
    assert len(raw_data) == segment_len * freq

    # Step 1: Divide the segment into fixed-length intervals, (i, s, c)
    interval_sensor_values = split_array_with_overlap(
        raw_data,
        INTERVAL_OVERLAP_RATIO,
        interval_len=int(INTERVAL_SPAN * freq),
    )

    # Step 2: Convert numpy array to tensor, and convert to [c. i, s] shape
    time_tensor = torch.from_numpy(interval_sensor_values).float()
    time_tensor = time_tensor.permute(2, 0, 1)

    return time_tensor


def process_one_sample(sample, labels, folder, time_output_path):
    """Process and save a sample.

    Args:
        sample (_type_): _description_
        folder (_type_): Contains labels and runs.
        shake (_type_): _description_
        output_path (_type_): _description_
    """
    id = sample["id"]
    time_output_file = os.path.join(time_output_path, f"{folder}_{id}.pt")

    if "speed" not in labels:
        label_dict = {
            "vehicle_type": torch.tensor(labels["vehicle"]).long(),
            "distance": torch.tensor(labels["distance"]).long(),
        }
    else:
        label_dict = {
            "vehicle_type": torch.tensor(labels["vehicle"]).long(),
            "speed": torch.tensor(labels["speed"]).long(),
            "distance": torch.tensor(labels["distance"]).long(),
        }

    time_sample = {
        "label": label_dict,
        "flag": {},
        "data": {},
    }

    # extract modality tensor
    for loc in sample["signal"]:
        # time placeholders
        time_sample["data"][loc] = dict()
        time_sample["flag"][loc] = dict()

        for mod in sample["signal"][loc]:
            time_tensor = extract_loc_mod_tensor(sample["signal"][loc][mod], SEGMENT_SPAN, FREQS[mod])

            # save time sample
            time_sample["data"][loc][mod] = time_tensor
            time_sample["flag"][loc][mod] = True

    # save the sample
    torch.save(time_sample, time_output_file)


def process_one_sample_wrapper(args):
    """Wrapper function for process a sample"""
    return process_one_sample(*args)


def process_one_folder(folder, labels, input_path, time_output_path):
    """Process a single mat file.

    Args:
        labels: {
            "vehicle": vehicle_type,
            "terrain": terrain_type,
            "speed": float(speed),
            "distance": float(distance),
        }
    """
    start_second = 2
    end_second = 1

    # Step 1: Loading original files
    print(f"Processing: {folder}")
    audio_file = os.path.join(input_path, folder, f"AUD_{folder}.csv")
    seismic_file = os.path.join(input_path, folder, f"EHZ_{folder}.csv")

    # load the audio, [channel, samples]
    raw_audio = np.loadtxt(audio_file, dtype=float, delimiter=" ")
    if raw_audio.ndim > 1:
        raw_audio = raw_audio[:, 0]
    raw_audio = np.expand_dims(raw_audio, axis=1)[1:]
    raw_audio = raw_audio[16000 * start_second : len(raw_audio) - 16000 * end_second]

    # load the seismic data
    raw_seismic = np.loadtxt(seismic_file, dtype=str, delimiter=" ")
    if raw_seismic.ndim > 1:
        raw_seismic = raw_seismic[:, 0].astype(float)
    raw_seismic = np.expand_dims(raw_seismic, axis=1)
    raw_seismic = raw_seismic[FREQS["seismic"] * start_second : len(raw_seismic) - FREQS["seismic"] * end_second]

    # extract the overlapped length
    length = min(len(raw_audio) // 16000, len(raw_seismic) // 100)
    raw_audio = raw_audio[: 16000 * length]
    raw_seismic = raw_seismic[: 100 * length]

    # downsample audio
    if AUD_DOWNSAMPLE_RATE > 1:
        raw_audio = resample_numpy_array(raw_audio, 16000, int(16000 / AUD_DOWNSAMPLE_RATE))

    # Step 2: Partition into individual samples
    splitted_data = {"audio": [], "seismic": []}
    splitted_data["audio"] = split_array_with_overlap(
        raw_audio,
        SEGMENT_OVERLAP_RATIO,
        interval_len=SEGMENT_SPAN * FREQS["audio"],
    )
    splitted_data["seismic"] = split_array_with_overlap(
        raw_seismic,
        SEGMENT_OVERLAP_RATIO,
        interval_len=SEGMENT_SPAN * FREQS["seismic"],
    )

    # prepare the individual samples
    sample_list = []
    for i in range(len(splitted_data["seismic"])):
        sample = {"id": i, "signal": {"shake": {}}}
        sample["signal"]["shake"]["audio"] = splitted_data["audio"][i]
        sample["signal"]["shake"]["seismic"] = splitted_data["seismic"][i]

        if len(splitted_data["seismic"][i]) < SEGMENT_SPAN * FREQS["seismic"]:
            continue
        else:
            sample_list.append(sample)

    # Step 3: Parallel processing and saving of the invidual samples
    print(f"Processing and saving individual samples: {folder, len(sample_list)}")
    pool = Pool(max_workers=cpu_count())
    args_list = [[sample, labels, folder, time_output_path] for sample in sample_list]
    pool.map(process_one_sample_wrapper, args_list, chunksize=1)
    pool.shutdown()


def process_one_folder_wrapper(args):
    """Wrapper function for procesing one folder."""
    return process_one_folder(*args)


if __name__ == "__main__":
    
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
        
    time_output_path = os.path.join(output_directory, "distance_speed_individual_time_samples")

    if not os.path.exists(time_output_path):
        os.mkdir(time_output_path)

    # list the files to process
    folders_to_process = []
    for e in os.listdir(input_path):
        if os.path.isdir(os.path.join(input_path, e)):
            folders_to_process.append(e)

    # process the files in parallel
    args_list = []
    for folder in folders_to_process:
        args_list.append([folder, parse_labels(folder), input_path, time_output_path])

    start = time.time()

    pool = Pool(max_workers=cpu_count())
    pool.map(process_one_folder_wrapper, args_list, chunksize=1)
    pool.shutdown()

    end = time.time()
    print("------------------------------------------------------------------------")
    print("Total execution time: %f s" % (end - start))
