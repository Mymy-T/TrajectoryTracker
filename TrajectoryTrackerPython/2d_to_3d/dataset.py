from enum import Enum
import numpy as np
import os
import pandas as pd
from projection_component_paths import Paths
from torch.utils.data import Dataset

class DatasetType(Enum):
    train = 0
    valid = 1
    test = 2

# TODO: Fix, adapt to the trajectory data; make a new data splitter.py

class TrajectoryReconstructionDataset(Dataset):

    def __init__(self, mode: DatasetType):
        paths = Paths()
        if mode == DatasetType.train:
            dataset_type = "train"
        elif mode == DatasetType.valid:
            dataset_type = "valid"
        else:
            dataset_type = "test"

        self.data_df = pd.read_csv(os.path.join(paths.dataset_csv_folder, f"{dataset_type}_data.csv"))
        self.data_folder = paths.synthetic_data_folder


    def __len__(self):
        return self.data_df.shape[0]
    

    def __getitem__(self, idx):
        selected_example = self.data_df.iloc[idx, :]
        trajectory_3d_path = os.path.join(self.data_folder, selected_example.loc["trajectory_3d_path"])
        trajectory_2d_path = os.path.join(self.data_folder, selected_example.loc["trajectory_2d_path"])
        
        trajectory_2d = np.load(trajectory_2d_path)
        trajectory_3d = np.load(trajectory_3d_path)

        return (trajectory_2d, trajectory_3d)
    
if __name__ == "__main__":
    data = TrajectoryReconstructionDataset(DatasetType.valid)
    print(f"Input shape: {data[0][0].shape}")
    print(f"Label shape: {data[0][1].shape}")
    print(f"Valid data num examples: {data.__len__()}")