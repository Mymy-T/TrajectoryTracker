import argparse
from dataset import collate_func, DatasetType, TrajectoryReconstructionDataset
import matplotlib.pyplot as plt
from model import TrajectoryReconstructionNetwork
import os
from paths import Paths
from settings import device
import torch
from torch.utils.data import DataLoader
from trainer import read_batch
from visualizer import plot_3d_trajectory


def main():
    parser = argparse.ArgumentParser(description="Run various types of inference.")
    parser.add_argument("--name", required=True, help="Name of the FULL model to run; don't include extensions")
    args = parser.parse_args()

    evaluate_single_example(args.name)


def evaluate_single_example(model_name):
    """
    model_name is just the name of the saved model folder, without any extensions or parent folders
    """

    data = TrajectoryReconstructionDataset(DatasetType.test, noise_2d=False)
    dataloader = DataLoader(data, collate_fn=collate_func, shuffle=False)
    example = next(iter(dataloader))

    model = TrajectoryReconstructionNetwork(saved_bounce_model_name=f"{model_name}_bm")
    saved_model_path = os.path.join(Paths().model_saves_folder, model_name, f"{model_name}.pt")
    state_dict = torch.load(saved_model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()

    with torch.inference_mode():
        inputs, coord_labels, eot_labels, bounce_token_labels, lengths, intrinsics, extrinsics, diameters = read_batch(example)

        pred_coords, pred_eot, pred_intermediate_heights = model(inputs, lengths, intrinsics, extrinsics, diameters) # (1, L, 3), (1, L, 1)

        pred_coords = pred_coords[0].cpu().numpy() # (L, 3)
        pred_eot = (torch.sigmoid(pred_eot[0, :, 0]) >= 0.5).float().cpu().numpy()  # (L,)
        coord_labels = coord_labels[0].cpu().numpy()
        eot_labels = eot_labels[0, :, 0].cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), subplot_kw={"projection": "3d"})

    plot_3d_trajectory(pred_coords, pred_eot, ax1, label="Prediction")
    plot_3d_trajectory(coord_labels, eot_labels, ax2, label="Actual")

    plot_3d_trajectory(coord_labels, eot_labels, ax3, color="b", label="Overlay", hide_legend=True)
    plot_3d_trajectory(pred_coords, pred_eot, ax3, color="orange", label="Overlay", hide_legend=True)


    plt.show()
    
if __name__ == "__main__":
    main()