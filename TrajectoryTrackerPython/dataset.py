import cv2
from enum import Enum
import numpy as np
import os
import pandas as pd
from paths import ProjectPaths
from torch.utils.data import Dataset

class DatasetType(Enum):
    train = 0
    valid = 1
    test = 2

# Based off of yastrebksv's implementation of TrackNet on github: https://github.com/yastrebksv/TrackNet/blob/main/datasets.py

class TrackNetDataset(Dataset):
    def __init__(self, mode: DatasetType, input_height=360, input_width=640):
        paths = ProjectPaths()
        if mode == DatasetType.train:
            dataset_type = "train"
        elif mode == DatasetType.valid:
            dataset_type = "valid"
        else:
            dataset_type = "test"
        self.data_df = pd.read_csv(os.path.join(paths.dataset_folder, f"{dataset_type}_data.csv"))       
        self.height = input_height
        self.width = input_width
        self.processed_data_folder = paths.processed_data_folder
        
    def __len__(self):
        return self.data_df.shape[0]
    
    def __getitem__(self, idx):
        #  self.data_df has columns [match_clip_id, heatmap(t), img(t), ...,  img(t-(frame_sequence_len - 1))]
        selected_paths = self.data_df.iloc[idx, 1:].apply(lambda x: os.path.join(self.processed_data_folder, x)).tolist()
        heatmap_path = selected_paths[0]
        img_paths = list(reversed(selected_paths[1:])) # [img(t-(frame_sequence_len-1)), ... , img(t-2), img(t-1), img(t)] -> order correct, where lower channel numbers are earlier timesteps
        inputs = self._get_input(img_paths)
        outputs = self._get_output(heatmap_path)
        return (inputs, outputs)
    
    def _get_output(self, heatmap_path):
        img = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
        # img = np.clip(img, 0, 255).astype(np.int32)
        # output = np.zeros((self.height, self.width, 256), dtype=np.float32) # Each pixel has a len 256 one-hot encoded vector; position of 1 corresponds to brightness value of pixel
        # grid_h, grid_w = np.ogrid[:self.height, :self.width] # grid_h is of shape (h, 1); grid_w is of shape (1, w)
        # output[grid_h, grid_w, img] = 1.0 # grid_h & grid_w broadcasted to (h, w); img is already (h, w); vectorized loop through all h*w indices
        # return output
        return img
        
    def _get_input(self, img_paths):
        imgs = []
        for img_path in img_paths:
            img = cv2.imread(img_path) # Keep in BGR format, the pretrained model was trained this way
            imgs.append(img) # Each img of shape (h, w, c)
        
        imgs = np.concatenate(imgs, axis=2) # Concatenate along channel dimension; lower channel dimensions correspond to earlier frames
        imgs = np.moveaxis(imgs, 2, 0) # Shape now (c, h, w)
        imgs = imgs.astype(np.float32) / 255.0 # Normalize pixel values to [0, 1] to enhance training stability
        return imgs
    
if __name__ == "__main__":
    data = TrackNetDataset(DatasetType.valid)
    print(f"Input shape: {data[0][0].shape}")
    print(f"Label shape: {data[0][1].shape}")
    print(f"Valid data num examples: {data.__len__()}")