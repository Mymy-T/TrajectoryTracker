from enum import Enum
import numpy as np
import os
import pandas as pd
from projection_component_paths import Paths
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class DatasetType(Enum):
    train = 0
    valid = 1
    test = 2


def homogenize_seq_length(batch):
    inputs, outputs = zip(*batch) # [(x1, y1), (x2, y2), ...] -> [(x1, x2, ...), (y1, y2, ...)]

    lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long) # Corresponding outputs have same sequence lengths
    
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    padded_outputs = pad_sequence(outputs, batch_first=True, padding_value=0.0)
    
    return padded_inputs, padded_outputs, lengths


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
        
        trajectory_2d = torch.from_numpy(np.load(trajectory_2d_path))
        trajectory_3d = torch.from_numpy(np.load(trajectory_3d_path))

        return (trajectory_2d, trajectory_3d)
    
if __name__ == "__main__":
    data = TrajectoryReconstructionDataset(DatasetType.valid)
    print(f"Input shape: {data[0][0].shape}")
    print(f"Label shape: {data[0][1].shape}")
    print("\n")
    print(f"Valid data num examples: {data.__len__()}")

    dataloader = DataLoader(data, batch_size=3, shuffle=True, collate_fn=homogenize_seq_length)
    batch = next(iter(dataloader))
    print("\n")
    print(f"Example batch input shape: {batch[0].shape}")
    print(f"Example batch output shape: {batch[1].shape}")
    print(f"Lengths array of above batch: {batch[2]}")
