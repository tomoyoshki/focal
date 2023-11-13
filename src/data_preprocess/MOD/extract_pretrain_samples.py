import os
import time

import numpy as np

from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import cpu_count
from extract_samples import process_one_sample_wrapper, resample_numpy_array, split_array_with_overlap, folder_to_label
from data_trunk import start_time_shift, end_time_shift
from preprocessing_config import output_directory, input_path as input_path

SEGMENT_SPAN = 2
INTERVAL_SPAN = 0.2
SEGMENT_OVERLAP_RATIO = 0.0
INTERVAL_OVERLAP_RATIO = 0.0
AUD_DOWNSAMPLE_RATE = 2


IDS = []

STD_THRESHOLD = 0

FREQS = {"audio": 16000 / AUD_DOWNSAMPLE_RATE, "seismic": 100, "acc": 100}
LABELS = {
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
PRESERVED_EXTRA_FOLDERS = {
    "motor": ["rs1", "rs2", "rs7"],
    "mustang0528": ["rs1", "rs2", "rs7"],
    "walk2": ["rs1", "rs2", "rs7"],
    "tesla": ["rs1", "rs2", "rs7"],
    "bicycle": ["rs1", "rs2", "rs3", "rs7"],
    "bicycle2": ["rs1", "rs2", "rs3", "rs7"],
    "forester": ["rs1", "rs2", "rs3", "rs7"],
    "forester2": ["rs1", "rs2", "rs3", "rs7"],
    "motor2": ["rs1", "rs2", "rs3", "rs7"],
    "pickup": ["rs1", "rs2", "rs3", "rs7"],
    "pickup2": ["rs1", "rs2", "rs3", "rs7"],
    "scooter": ["rs1", "rs2", "rs3", "rs7"],
    "scooter2": ["rs1", "rs2", "rs3", "rs7"],
    "walk": ["rs1", "rs2", "rs3", "rs7"],
}


def process_one_shake(
    folder,
    shake,
    label_id,
    input_path,
    freq_output_path,
    time_output_path,
):
    """Process a single folder.

    Args:
        input_folder (_type_): _description_
        output_path (_type_): _description_
    """
    # Step 1: Loading original files
    shake_path = os.path.join(input_path, folder, shake)

    start_second = start_time_shift[folder][shake]
    end_second = end_time_shift[folder][shake]

    print(f"Processing: {shake_path}")

    # load the audio, the file names are different for data collected from different places
    if "aud16000.csv" in os.listdir(shake_path):
        audio_file = "aud16000.csv"
    else:
        audio_file = "aud.csv"
    raw_audio = np.loadtxt(os.path.join(shake_path, audio_file), dtype=float, delimiter=",")
    if raw_audio.ndim > 1:
        raw_audio = raw_audio[:, 0]
    raw_audio = np.expand_dims(raw_audio, axis=1)
    raw_audio = raw_audio[16000 * start_second : len(raw_audio) - 16000 * end_second]

    # resample the audio
    if AUD_DOWNSAMPLE_RATE > 1:
        raw_audio = resample_numpy_array(raw_audio, 16000, int(16000 / AUD_DOWNSAMPLE_RATE))

    # load the seismic data
    raw_seismic = np.loadtxt(os.path.join(shake_path, "ehz.csv"), dtype=float, delimiter=" ")
    if raw_seismic.ndim > 1:
        raw_seismic = raw_seismic[:, 0]
    raw_seismic = np.expand_dims(raw_seismic, axis=1)
    raw_seismic = raw_seismic[FREQS["seismic"] * start_second : len(raw_seismic) - FREQS["seismic"] * end_second]
    print(folder, shake, np.shape(raw_audio), np.shape(raw_seismic))

    # Step 2: Partition into individual samples
    splitted_data = {"audio": [], "seismic": [], "acc": []}
    splitted_data["audio"] = split_array_with_overlap(
        raw_audio, SEGMENT_OVERLAP_RATIO, interval_len=SEGMENT_SPAN * FREQS["audio"]
    )
    splitted_data["seismic"] = split_array_with_overlap(
        raw_seismic, SEGMENT_OVERLAP_RATIO, interval_len=SEGMENT_SPAN * FREQS["seismic"]
    )

    # prepare the individual samples
    sample_list = []
    for i in range(len(splitted_data["seismic"])):
        # Filter the data of parkpand according to their std
        if label_id not in [0, 1, 2]:
            if np.std(splitted_data["seismic"][i]) < STD_THRESHOLD:
                continue

        
        sample = {"id": i, "signal": {"shake": {}}}
        sample["signal"]["shake"]["audio"] = splitted_data["audio"][i]
        sample["signal"]["shake"]["seismic"] = splitted_data["seismic"][i]

        if len(splitted_data["seismic"][i]) < SEGMENT_SPAN * FREQS["seismic"]:
            continue
        else:
            IDS.append(i)
            sample_list.append(sample)
    print(len(IDS))
    # Step 3: Parallel processing and saving of the invidual samples
    print(f"Processing and saving individual samples: {folder, shake, len(sample_list)}")
    pool = Pool(max_workers=cpu_count())
    args_list = [[sample, label_id, folder, shake, freq_output_path, time_output_path] for sample in sample_list]
    pool.map(process_one_sample_wrapper, args_list, chunksize=1)
    pool.shutdown()


def process_one_shake_wrapper(args):
    """Wrapper function for procesing one folder."""
    return process_one_shake(*args)


if __name__ == "__main__":
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    time_output_path = os.path.join(output_directory, "extra_time_samples")
    freq_output_path = os.path.join(output_directory, "extra_freq_samples")

    for f in [freq_output_path, time_output_path]:
        if not os.path.exists(f):
            os.mkdir(f)

    folders_to_process = []
    for e in os.listdir(input_path):
        if e in PRESERVED_EXTRA_FOLDERS:
            folders_to_process.append(e)

    # extract args list
    args_list = []
    for folder in folders_to_process:
        shake_list = os.listdir(os.path.join(input_path, folder))
        label, label_id = folder_to_label(folder)
        for shake in PRESERVED_EXTRA_FOLDERS[folder]:
            args_list.append([folder, shake, label_id, input_path, freq_output_path, time_output_path])

    start = time.time()

    pool = Pool(max_workers=cpu_count())
    pool.map(process_one_shake_wrapper, args_list, chunksize=1)
    pool.shutdown()

    end = time.time()
    print("------------------------------------------------------------------------")
    print("Total execution time: %f s" % (end - start))