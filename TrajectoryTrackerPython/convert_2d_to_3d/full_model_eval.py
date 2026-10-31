import argparse
from convert_2d_to_3d.dataset import TrajectoryReconstructionDataset, DatasetType, collate_func
from convert_2d_to_3d.loss import FullLoss, TrajectoryLoss
from convert_2d_to_3d.model import TrajectoryReconstructionNetwork
from convert_2d_to_3d.paths import Paths
from convert_2d_to_3d import settings
from convert_2d_to_3d.trainer import evaluate, read_batch
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Evaluate the loss metrics of a full trajectory reconstruction network.")
    parser.add_argument("--name", required=True, help="The name of the model. This should be a simple string, without extensions.")
    parser.add_argument("--true-bounces", action="store_true", help="If included, evaluate utilizing the true bounce tokens instead of the predicted bounce tokens.")
    parser.add_argument("--no-noise", action="store_true", help="If included, the model will be evaluated without noise in the trajectory.")
    parser.add_argument("--mse-histogram", action="store_true", help="Evaluate by plotting a histogram of the trajectory MSE values.")
    
    args = parser.parse_args()

    name = args.name
    model_path = os.path.join(Paths().model_saves_folder, f"{name}", f"{name}.pt")
    model = TrajectoryReconstructionNetwork(saved_bounce_model_name=f"{name}_bm")
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(settings.device)

    include_noise = not args.no_noise
    test_data = TrajectoryReconstructionDataset(DatasetType.test, noise_2d=include_noise)
    running_on_cpu = (settings.device.type == "cpu")
    test_dataloader = DataLoader(test_data, batch_size=settings.Training().train_config.batch_size, pin_memory=(not running_on_cpu), collate_fn=collate_func)


    if not args.mse_histogram:
        train_config = settings.Training().train_config
        loss_fn = FullLoss(eot_imbalance_weight=train_config.eot_imbalance_weight, loss_weights=train_config.loss_component_weights)
        evaluate(model, loss_fn, test_dataloader, valid=False, true_bounces=args.true_bounces, verbose=True)
    else:
        loss_fn = TrajectoryLoss(return_vector=True)
        plot_trajectory_error_histogram(model, loss_fn, test_dataloader, true_bounces=args.true_bounces)


def plot_trajectory_error_histogram(model: TrajectoryReconstructionNetwork, loss_fn, data_loader, true_bounces):    

    model.eval()

    mse_list = []
    
    with torch.inference_mode():

        for batch in tqdm(data_loader):
            inputs, coord_labels, eot_labels, bounce_token_labels, lengths, intrinsics, extrinsics, diameters = read_batch(batch)

            if true_bounces:
                pred_coords, pred_eot, pred_intermediate_heights = model(inputs, lengths, intrinsics, extrinsics, diameters, bounce_token_labels) # (N, L, 3), (N, L, 1)
            else:
                pred_coords, pred_eot, pred_intermediate_heights = model(inputs, lengths, intrinsics, extrinsics, diameters) # (N, L, 3), (N, L, 1)
            
            mse_vec = loss_fn(pred_coords, coord_labels, lengths) # (N,)
            mse_list += mse_vec.tolist()
        
    print(f"len(mse_list) = {len(mse_list)}")
    n_bins = floor(len(mse_list) / 30)
    print(f"n_bins = {n_bins}")
    
    plt.figure(figsize=(10, 6))

    plt.hist(mse_list, bins=n_bins)
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.title("MSE Histogram")

    p0, p25, p50, p75, p100 = np.percentile(np.array(mse_list), [0, 25, 50, 75, 100])
    summary_text = f"Min: {p0:.4f}   |   25th Percentile: {p25:.4f}   |   Median (50th): {p50:.4f}   |   75th Percentile: {p75:.4f}   |   Max: {p100:.4f}"
    num_text = f"Num Examples: {len(mse_list)}   |   Num Bins: {n_bins}"
    plt.figtext(0.5, 0.017, summary_text, ha="center")
    plt.figtext(0.05, 0.9, num_text)

    plt.show()

    
if __name__ == "__main__":
    main()