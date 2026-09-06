from enum import Enum
import numpy as np
import os
import pandas as pd
from paths import Paths
import settings
from statistics import NormalDist
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class DatasetType(Enum):
    train = 0
    valid = 1
    test = 2


def collate_func(batch):
    sequences = [batch[i][0] for i in range(len(batch))] # [(i1, o1, el1), (i2, o2, el2), ...]
    cam_configs = [batch[i][1] for i in range(len(batch))] # [(intrinsics_1, extrinsics_1), (intrinsics_2, extrinsics_2), ...]
    diameters = [batch[i][2] for i in range(len(batch))] # [diameters_1, diameters_2, ...]

    inputs, outputs, eot_labels = zip(*sequences) # [(i1, o1, el1), (i2, o2, el2), ...] -> [(i1, i2, ...), (o1, o2, ...), (el1, el2, ...)]
    intrinsics, extrinsics = zip(*cam_configs) # [(intrinsics_1, extrinsics_1), (intrinsics_2, extrinsics_2), ...] -> [(intrinsics_1, intrinsics_2, ...), (extrinsics_1, extrinsics_2, ...)]
    
    intrinsics = torch.stack(intrinsics) # (N, 3, 3)
    extrinsics = torch.stack(extrinsics) # (N, 4, 4)

    diameters = torch.stack(diameters) # (N, 2)

    lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long) # Corresponding outputs have same sequence lengths; shape of lengths is (N,)
    
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    padded_outputs = pad_sequence(outputs, batch_first=True, padding_value=0.0)
    padded_eot_labels = pad_sequence(eot_labels, batch_first=True, padding_value=0.0)
    
    return padded_inputs, padded_outputs, padded_eot_labels, lengths, intrinsics, extrinsics, diameters


def obtain_intrinsics(fx, fy, cx, cy):
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return intrinsics

def obtain_extrinsics(extrinsic_elements: list):

    if len(extrinsic_elements) != 16:
        raise ValueError("extrinsics could not be obtained for example")
    
    row_1 = extrinsic_elements[0:4]
    row_2 = extrinsic_elements[4:8]
    row_3 = extrinsic_elements[8:12]
    row_4 = extrinsic_elements[12:16]

    extrinsics = np.array([row_1, row_2, row_3, row_4], dtype=np.float64)
    return extrinsics


class TrajectoryReconstructionDataset(Dataset):

    def __init__(self, mode: DatasetType, noise_2d: bool, noise_bound=10.0, within_noise_bound_prob=0.95, seed=0):
        paths = Paths()
        if mode == DatasetType.train:
            dataset_type = "train"
        elif mode == DatasetType.valid:
            dataset_type = "valid"
        else:
            dataset_type = "test"

        self.data_df = pd.read_csv(os.path.join(paths.dataset_csv_folder, f"{dataset_type}_data.csv"))
        self.data_folder = paths.synthetic_data_folder

        self.noise_2d = noise_2d
        self.rng = np.random.default_rng(seed) # Used for adding random noise
        std_norm_dist = NormalDist(0, 1)
        p = within_noise_bound_prob/2 + 0.5
        self.sigma = noise_bound / std_norm_dist.inv_cdf(p)



    def __len__(self):
        return self.data_df.shape[0]
    

    def __getitem__(self, idx):
        selected_example = self.data_df.iloc[idx, :]

        fx, fy, cx, cy = selected_example.loc[["fx", "fy", "cx", "cy"]]

        extrinsics_col_names = [f"extrinsics_{i}" for i in range(16)]
        extrinsic_elements = selected_example.loc[extrinsics_col_names].to_list()

        intrinsics = torch.from_numpy(obtain_intrinsics(fx, fy, cx, cy)).to(dtype=settings.torch_dtype) # (3, 3)
        extrinsics = torch.from_numpy(obtain_extrinsics(extrinsic_elements)).to(dtype=settings.torch_dtype) # (4, 4)

        trajectory_3d_path = os.path.join(self.data_folder, selected_example.loc["trajectory_3d_path"])
        trajectory_2d_path = os.path.join(self.data_folder, selected_example.loc["trajectory_2d_path"])
        trajectory_eot_path = os.path.join(self.data_folder, selected_example.loc["trajectory_eot_path"])
        trajectory_diameter_path = os.path.join(self.data_folder, selected_example.loc["trajectory_diameter_path"])
        
        trajectory_2d = torch.from_numpy(np.load(trajectory_2d_path)).to(dtype=settings.torch_dtype) # (L, 2)
        trajectory_3d = torch.from_numpy(np.load(trajectory_3d_path)).to(dtype=settings.torch_dtype) # (L, 3)
        trajectory_eot = torch.from_numpy(np.load(trajectory_eot_path)).unsqueeze(-1).to(dtype=settings.torch_dtype) # (L, 1)
        trajectory_diameters = torch.from_numpy(np.load(trajectory_diameter_path)).to(dtype=settings.torch_dtype) # (2,)

        if self.noise_2d:
            trajectory_noise_2d = torch.from_numpy(self.rng.normal(loc=0.0, scale=self.sigma, size=trajectory_2d.shape)).to(dtype=settings.torch_dtype)
            trajectory_2d += trajectory_noise_2d


        return (trajectory_2d, trajectory_3d, trajectory_eot), (intrinsics, extrinsics), trajectory_diameters
    
if __name__ == "__main__":
    data = TrajectoryReconstructionDataset(DatasetType.valid, noise_2d=False) # Data has shape (N, x, y) where y = 3 if x = 1 and y = 2 if x = 2
    print(f"Input shape: {data[0][0][0].shape}")
    print(f"Label shape: {data[0][0][1].shape}")
    print("\n")
    print(f"Num examples in dataset: {data.__len__()}")

    dataloader = DataLoader(data, batch_size=3, shuffle=True, collate_fn=collate_func) # Batch has shape (7, N, L, D)
    batch = next(iter(dataloader))
    print("\n")
    print(f"Batch number of variables: {len(batch)}")
    print(f"Example batch 2d coords shape: {batch[0].shape}")
    print(f"Example batch 3d coords shape: {batch[1].shape}")
    print(f"EOT array of above batch: {batch[2].shape}")
    print(f"Lengths array: {batch[3]}")
    print(f"Intrinsics: {batch[4].shape}")
    print(f"Extrinsics: {batch[5].shape}")
    print(f"Initial Heights: {batch[6].shape}")
