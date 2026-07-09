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
    processed_data_path is the path of the processed data
    mode is one of the options defined within DataSplitMode, and defines split boundaries
    frame_sequence_len is the number of frames; must be >= 1

    creates new folders called train_data, valid_data, & test_data; sizes are based on train_prop and valid_prop, and are approximate when mode != no_border
    each folder has a .csv referencing the appropriate img/heatmap in the processed_data folder
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
    cols = ["match_id", "clip_id", "heatmap(t)", "img(t)"]
    if frame_sequence_len > 1:
        additional_cols = [f"img(t-{i})" for i in range(1, frame_sequence_len)]
        cols = cols + additional_cols
    # df_idx = pd.Index([], name="id")
    # df = pd.DataFrame(columns=cols, index=idx)
    dataset_parts = []
    matches = sorted([match for match in os.listdir(processed_data_path) if not match.startswith(".")]) # All matches have a name of the form match_####; ignore hidden files such as .DS_Store
    for match in matches:
        match_path = os.path.join(processed_data_path, match)
        clips = sorted([clip for clip in os.listdir(match_path) if not clip.startswith(".")]) # All clips have a name of the form clip_####; ignore hidden files such as .DS_Store
        for clip in clips:
            clip_folder = os.path.join(match_path, clip)
            img_folder = os.path.join(clip_folder, "images")
            heatmap_folder = os.path.join(clip_folder, "heatmaps")
            T = len(os.listdir(img_folder))
            unstacked_frame_nums = [[j for j in range(i, T - frame_sequence_len + 1 + i)] for i in range(frame_sequence_len)] # See d2l.ai, section 9.1.3; creates sequences of frame_sequence_len
            unstacked_frame_nums += [[j for j in range(frame_sequence_len - 1, T)]] # Duplicate last element so that it becomes the column for the ground truth heatmap path when stacked
            frame_nums = np.stack(unstacked_frame_nums, axis=1)
            convert_num_to_img_path = np.vectorize(lambda x: os.path.join(img_folder, f"frame_{x:08d}.jpg")) # Name format is by convention in this project
            convert_num_to_heatmap_path = np.vectorize(lambda x: os.path.join(heatmap_folder, f"frame_{x:08d}_heatmap.png")) # Name format is by convention in this project
            img_paths = convert_num_to_img_path(frame_nums[:, :-1])
            heatmap_paths = convert_num_to_heatmap_path(frame_nums[:, -1:]) # Colon after -1 to prevent dimension collapse (don't drop column dimension, that's needed for concatenation)
            dataset_part = np.concatenate((img_paths, heatmap_paths), axis=1) # dataset_part has columns starting with  t-(frame_sequence_len - 1) up to img(t), and then heatmap(t); columns will be reversed outside the loop
            dataset_part = np.insert(dataset_part, dataset_part.shape[1], clip, axis=1) # Add clip id
            dataset_part = np.insert(dataset_part, dataset_part.shape[1], match, axis=1) # Add match id
            dataset_parts.append(dataset_part) # Final dataset_part has columns starting with  t-(frame_sequence_len - 1) up to img(t), then heatmap(t), clip_id, and match_id; will be reversed to match cols

    dataset = np.flip(np.concatenate(dataset_parts, axis=0), axis=1) # Combine all the parts, and then reverse the columns of dataset to get it to match the dataframe
    # temp_df = pd.DataFrame(dataset, columns=cols)
    # df = pd.concat([df, temp_df], ignore_index=True)
    df = pd.DataFrame(dataset, columns=cols)
    return df


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
    pass # TODO

def _split_data_match_borders(df, train_prop, valid_prop, out_path):
    pass # TODO

if __name__ == "__main__":
    paths = ProjectPaths()
    mode = DataSplitMode.no_border
    split_data(processed_data_path=paths.processed_data_folder, out_path=paths.dataset_folder, mode=mode, train_prop=0.7, valid_prop=0.15)