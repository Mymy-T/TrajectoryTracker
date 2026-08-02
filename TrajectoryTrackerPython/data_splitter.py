from enum import Enum
import math
import numpy as np
import os
import pandas as pd
from paths import ProjectPaths

class DataSplitMode(Enum):
    match_border = 0 # Split only at match-level
    clip_border = 1 # Split only at clip-level
    no_border = 2 # Split regardless of match/clip level

def split_data(processed_data_path, out_path, mode, train_prop, valid_prop, frame_sequence_len=3):
    """
    - frame_sequence_len >= 1
    - Sizes of train, valid, and test datasets are approximate when mode != no_border
    """

    if not frame_sequence_len >= 1:
        print("Error: frame_sequence_len must be greater than or equal to 1.")
        return

    data_df = _create_data_df(processed_data_path, frame_sequence_len)
    
    if mode == DataSplitMode.match_border:
        _split_data_match_borders(data_df, train_prop, valid_prop, out_path)
    elif mode == DataSplitMode.clip_border:
        _split_data_clip_borders(data_df, train_prop, valid_prop, out_path)
    else:
        _split_data_no_borders(data_df, train_prop, valid_prop, out_path)
    
    print("Split success!")
    print(f"See {out_path}")

def _create_data_df(processed_data_path, frame_sequence_len):
    # Initialize the dataframe containing all the paths
    cols = ["match_clip_id", "heatmap(t)", "img(t)"]
    if frame_sequence_len > 1:
        additional_cols = [f"img(t-{i})" for i in range(1, frame_sequence_len)]
        cols = cols + additional_cols
    dataset_parts = []
    matches = sorted([match for match in os.listdir(processed_data_path) if not match.startswith(".")]) # All matches have a name of the form match_####; ignore hidden files such as .DS_Store
    for match in matches:
        match_path = os.path.join(processed_data_path, match)
        clips = sorted([clip for clip in os.listdir(match_path) if not clip.startswith(".")]) # All clips have a name of the form clip_####; ignore hidden files such as .DS_Store
        for clip in clips:
            clip_folder = os.path.join(match_path, clip)
            img_folder = os.path.join(clip_folder, "images")
            T = len(os.listdir(img_folder))

            missing_frame_nums = []
            for i in range(T): # Check for frames that were skipped when dataset_creator.py was run
                processed_img_path = os.path.join(img_folder, f"frame_{i:08d}.jpg") # label could be used too, it doesn't matter here
                if not os.path.exists(processed_img_path):
                    missing_frame_nums.append(i)

            unstacked_frame_nums = [[j for j in range(i, T - frame_sequence_len + 1 + i)] for i in range(frame_sequence_len)] # See d2l.ai, section 9.1.3; creates sequences of frame_sequence_len
            unstacked_frame_nums += [[j for j in range(frame_sequence_len - 1, T)]] # Duplicate last element so that it becomes the column for the ground truth heatmap path when stacked
            frame_nums = np.stack(unstacked_frame_nums, axis=1)
            # NOTE: The paths saved in the csv files are relative paths; when loading them in dataset.py, join them with processed_data_path
            convert_num_to_img_path = np.vectorize(lambda x: os.path.join(match, clip, "images", f"frame_{x:08d}.jpg")) 
            convert_num_to_heatmap_path = np.vectorize(lambda x: os.path.join(match, clip, "heatmaps", f"frame_{x:08d}_heatmap.png"))
            img_paths = convert_num_to_img_path(frame_nums[:, :-1])
            heatmap_paths = convert_num_to_heatmap_path(frame_nums[:, -1:]) # Colon after -1 to prevent dimension collapse (don't drop column dimension, that's needed for concatenation)
            dataset_part = np.concatenate((img_paths, heatmap_paths), axis=1) # dataset_part has columns starting with  t-(frame_sequence_len - 1) up to img(t), and then heatmap(t); columns will be reversed outside the loop
            match_clip_id = f"{match}__{clip}"
            dataset_part = np.insert(dataset_part, dataset_part.shape[1], match_clip_id, axis=1)

            if len(missing_frame_nums) != 0:
                missing_frames = convert_num_to_img_path(missing_frame_nums) # Use img paths, as this will auto drop anywhere an invalid image appears, including when the label is also invalid (which occurs when the invalid image is at timestep t)
                missing_frame_mask = np.isin(dataset_part, missing_frames).any(axis=1)
                dataset_part = dataset_part[~missing_frame_mask] # Ignore invalid images/labels

            dataset_parts.append(dataset_part) # Final dataset_part has columns starting with  t-(frame_sequence_len - 1) up to img(t), then heatmap(t), then match_clip_id; will be reversed to match cols

    dataset = np.flip(np.concatenate(dataset_parts, axis=0), axis=1) # Combine all the parts, and then reverse the columns of dataset to get it to match the dataframe
    df = pd.DataFrame(dataset, columns=cols)
    return df

def _generate_folder_level_split(ideal_num, prev_folder_end_index, folder_indices, folder_dict):
    end_id = prev_folder_end_index
    curr_sum = 0
    for i in range(prev_folder_end_index, len(folder_indices)): # Won't run if prev_clip_end_id >= len(clip_ids)
        old_sum = curr_sum
        curr_sum += folder_dict[folder_indices[i]]
        old_dist = abs(old_sum - ideal_num)
        curr_dist = abs(curr_sum - ideal_num)
        if old_dist <= curr_dist: # less than or equal to means that if adding another clip will result in overshooting ideal_num by the same amount it was missing, it will prefer to skip and leave an undersized dataset
            end_id = i
            break
        if i == len(folder_indices) - 1: # If we never break out of the loop
            end_id = len(folder_indices)
    
    return end_id # end_id is not included in slice


def _split_data_no_borders(df, train_prop, valid_prop, out_path):
    # Splitting into train/valid/test
    N = df.shape[0]
    train_num = math.floor(train_prop * N)
    valid_num = math.floor(valid_prop * N)
    train_df = df.iloc[:train_num, :]
    valid_df = df.iloc[train_num:train_num+valid_num, :]
    test_df = df.iloc[train_num+valid_num:, :]
    train_df.to_csv(os.path.join(out_path, "train_data.csv"), index=False)
    valid_df.to_csv(os.path.join(out_path, "valid_data.csv"), index=False)
    test_df.to_csv(os.path.join(out_path, "test_data.csv"), index=False)

def _split_data_clip_borders(df, train_prop, valid_prop, out_path):
    N = df.shape[0]
    train_num = math.floor(train_prop * N)
    valid_num = math.floor(valid_prop * N)
    clip_dict = df["match_clip_id"].value_counts().to_dict()
    clip_ids = sorted(list(clip_dict.keys())) # Each key is of the form match_####__clip_####; sorts by match # first, then clip #

    train_end_i = _generate_folder_level_split(ideal_num=train_num, prev_folder_end_index=0, folder_indices=clip_ids, folder_dict=clip_dict)
    valid_end_i = _generate_folder_level_split(ideal_num=valid_num, prev_folder_end_index=train_end_i, folder_indices=clip_ids, folder_dict=clip_dict)

    if train_end_i == 0:
        print("Not enough clips; train data is empty")

    if train_end_i == valid_end_i:
        print("Not enough clips; valid data is empty")

    if valid_end_i == len(clip_ids):
        print("Not enough clips; test data is empty")

    train_clip_ids = clip_ids[:train_end_i]
    valid_clip_ids = clip_ids[train_end_i:valid_end_i]
    test_clip_ids = clip_ids[valid_end_i:]

    train_df = df[df["match_clip_id"].isin(train_clip_ids)]
    valid_df = df[df["match_clip_id"].isin(valid_clip_ids)]
    test_df = df[df["match_clip_id"].isin(test_clip_ids)]

    train_df.to_csv(os.path.join(out_path, "train_data.csv"), index=False)
    valid_df.to_csv(os.path.join(out_path, "valid_data.csv"), index=False)
    test_df.to_csv(os.path.join(out_path, "test_data.csv"), index=False)



def _split_data_match_borders(df, train_prop, valid_prop, out_path):
    N = df.shape[0]
    train_num = math.floor(train_prop * N)
    valid_num = math.floor(valid_prop * N)
    df_copy = df.copy()
    df_copy["match_id"] = df["match_clip_id"].str.split('__', n=1).str[0]
    match_dict = df_copy["match_id"].value_counts().to_dict()
    match_ids = sorted(list(match_dict.keys())) # Each key is of the form match_####__clip_####; sorts by match # first, then clip #

    train_end_i = _generate_folder_level_split(ideal_num=train_num, prev_folder_end_index=0, folder_indices=match_ids, folder_dict=match_dict)
    valid_end_i = _generate_folder_level_split(ideal_num=valid_num, prev_folder_end_index=train_end_i, folder_indices=match_ids, folder_dict=match_dict)

    if train_end_i == 0:
        print("Not enough clips; train data is empty")

    if train_end_i == valid_end_i:
        print("Not enough clips; valid data is empty")

    if valid_end_i == len(match_ids):
        print("Not enough clips; test data is empty")

    train_match_ids = match_ids[:train_end_i]
    valid_match_ids = match_ids[train_end_i:valid_end_i]
    test_match_ids = match_ids[valid_end_i:]

    train_df = df[df_copy["match_id"].isin(train_match_ids)]
    valid_df = df[df_copy["match_id"].isin(valid_match_ids)]
    test_df = df[df_copy["match_id"].isin(test_match_ids)]

    train_df.to_csv(os.path.join(out_path, "train_data.csv"), index=False)
    valid_df.to_csv(os.path.join(out_path, "valid_data.csv"), index=False)
    test_df.to_csv(os.path.join(out_path, "test_data.csv"), index=False)


if __name__ == "__main__":
    paths = ProjectPaths()
    mode = DataSplitMode.no_border
    split_data(processed_data_path=paths.processed_data_folder, out_path=paths.dataset_folder, mode=mode, train_prop=0.7, valid_prop=0.15)