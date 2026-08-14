from enum import Enum
import math
import os
import pandas as pd
from projection_component_paths import Paths

class DataSplitMode(Enum):
    camera_config_border = 0 # Split only at different camera configs
    no_border = 1 # Split anywhere

def split_data(dataset_folder, out_path, mode, train_prop, valid_prop, seed=0):
    """
    - dataset_folder must contain dataset.csv (written by dataset_creator.py), with one row per (camera config, trajectory) example
    - Sizes of train, valid, and test datasets are approximate when mode != no_border
    """

    data_df = pd.read_csv(os.path.join(dataset_folder, "dataset.csv"))
    os.makedirs(out_path, exist_ok=True)

    if mode == DataSplitMode.camera_config_border:
        _split_data_camera_config_borders(data_df, train_prop, valid_prop, out_path)
    else:
        _split_data_no_borders(data_df, train_prop, valid_prop, out_path, seed)

    print("Split success!")
    print(f"See {out_path}")


def _generate_folder_level_split(ideal_num, prev_folder_end_index, folder_indices, folder_dict):
    end_id = prev_folder_end_index
    curr_sum = 0
    for i in range(prev_folder_end_index, len(folder_indices)): # Won't run if prev_clip_end_id >= len(clip_ids)
        old_sum = curr_sum
        curr_sum += folder_dict[folder_indices[i]]
        old_dist = abs(old_sum - ideal_num)
        curr_dist = abs(curr_sum - ideal_num)
        if old_dist <= curr_dist: # Less than or equal to means that if adding another clip will result in overshooting ideal_num by the same amount it was missing, it will prefer to skip and leave an undersized dataset
            end_id = i
            break
        if i == len(folder_indices) - 1: # If we never break out of the loop (otherwise end_id will be undefined)
            end_id = len(folder_indices)

    return end_id # end_id is not included in slice


def _split_data_no_borders(df, train_prop, valid_prop, out_path, seed):
    # Shuffle first to remove the remnant grouping by camera config
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    N = df.shape[0]
    train_num = math.floor(train_prop * N)
    valid_num = math.floor(valid_prop * N)
    train_df = df.iloc[:train_num, :]
    valid_df = df.iloc[train_num:train_num+valid_num, :]
    test_df = df.iloc[train_num+valid_num:, :]
    train_df.to_csv(os.path.join(out_path, "train_data.csv"), index=False)
    valid_df.to_csv(os.path.join(out_path, "valid_data.csv"), index=False)
    test_df.to_csv(os.path.join(out_path, "test_data.csv"), index=False)


def _split_data_camera_config_borders(df, train_prop, valid_prop, out_path):
    N = df.shape[0]
    train_num = math.floor(train_prop * N)
    valid_num = math.floor(valid_prop * N)
    config_dict = df["camera_config_id"].value_counts().to_dict()
    config_ids = sorted(config_dict.keys())

    train_end_i = _generate_folder_level_split(ideal_num=train_num, prev_folder_end_index=0, folder_indices=config_ids, folder_dict=config_dict)
    valid_end_i = _generate_folder_level_split(ideal_num=valid_num, prev_folder_end_index=train_end_i, folder_indices=config_ids, folder_dict=config_dict)

    if train_end_i == 0:
        print("Not enough camera configs; train data is empty")

    if train_end_i == valid_end_i:
        print("Not enough camera configs; valid data is empty")

    if valid_end_i == len(config_ids):
        print("Not enough camera configs; test data is empty")

    train_config_ids = config_ids[:train_end_i]
    valid_config_ids = config_ids[train_end_i:valid_end_i]
    test_config_ids = config_ids[valid_end_i:]

    train_df = df[df["camera_config_id"].isin(train_config_ids)]
    valid_df = df[df["camera_config_id"].isin(valid_config_ids)]
    test_df = df[df["camera_config_id"].isin(test_config_ids)]

    train_df.to_csv(os.path.join(out_path, "train_data.csv"), index=False)
    valid_df.to_csv(os.path.join(out_path, "valid_data.csv"), index=False)
    test_df.to_csv(os.path.join(out_path, "test_data.csv"), index=False)


if __name__ == "__main__":
    paths = Paths()
    mode = DataSplitMode.camera_config_border
    split_data(dataset_folder=paths.synthetic_data_folder, out_path=paths.dataset_csv_folder, mode=mode, train_prop=0.7, valid_prop=0.15)
