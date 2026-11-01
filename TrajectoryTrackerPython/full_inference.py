# TODO: Run full model; use a clip in the base project path -> run ball detection inference -> create heatmap and annotated video (save them) -> extract 2D coordinates, load intrinsics/extrinsics based on a sample img with keypoints -> 2d to 3d model -> plot 3d trajectory

import argparse
from ball_detection.frame_extractor import extract_clip_frames
from ball_detection.inference import create_annotated_video, create_heatmap_video, obtain_tracked_ball_data
from convert_2d_to_3d.inference import run_model, plot
from convert_2d_to_3d.projector import obtain_base_intrinsics, obtain_base_extrinsics
import numpy as np
import os
import pandas as pd
import torch

# python3 full_inference.py --dm pretrained_model --fm first_trial

def main():

    parser = argparse.ArgumentParser(description="Run a full inference pipeline")
    parser.add_argument("--dm", required=True, help="Name of the ball detection model. Don't add file extensions.")
    parser.add_argument("--fm", required=True, help="Name of the full 2d to 3D model. Don't add file extensions.")
    parser.add_argument("--video", default="video", help="Name of video to analyze within full_inference_results. Don't add extension.")
    parser.add_argument("--plot", action="store_true", help="Plot the predicted trajectory immediately afterwards.")
    args = parser.parse_args()

    inference_base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_inference_results")
    if not os.path.exists(inference_base_folder):
        os.mkdir(inference_base_folder)

    pred_coords, pred_eot, frame_indices = run_inference(args, inference_base_folder)
    save_trajectory_3d(pred_coords, pred_eot, frame_indices, inference_base_folder)

    if args.plot:
        plot(pred_coords, pred_eot)


def run_inference(args, inference_base_folder):
    frames_folder = os.path.join(inference_base_folder, "frames")
    if not os.path.exists(frames_folder):
        os.mkdir(frames_folder)

    video_path = os.path.join(inference_base_folder, f"{args.video}.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError("Couldn't find video path within full_inference_results.")

    intrinsics_path = os.path.join(inference_base_folder, f"intrinsics.json")
    if not os.path.exists(intrinsics_path):
        raise FileNotFoundError("Couldn't find intrinsics path within full_inference_results.")

    extrinsics_path = os.path.join(inference_base_folder, f"extrinsics.npy")
    if not os.path.exists(extrinsics_path):
        raise FileNotFoundError("Couldn't find extrinsics path within full_inference_results.")

    extract_clip_frames(video_path=video_path, output_dir=frames_folder)
    tracked_ball_points = obtain_tracked_ball_data(inference_video_folder=inference_base_folder, model_name=args.dm) # Dict of {frame_idx: [x, y, r]}
    create_heatmap_video(inference_video_folder=inference_base_folder)
    create_annotated_video(inference_video_folder=inference_base_folder, tracked_ball_points=tracked_ball_points)

    max_idx = max(list(tracked_ball_points.keys()))

    ball_points_df = pd.DataFrame(columns=["x", "y", "d"], dtype=float)
    ball_points_df = ball_points_df.reindex(range(max_idx + 1))

    for i in tracked_ball_points.keys():
        ball_points_df.loc[i, "x"] = tracked_ball_points[i][0] # x
        ball_points_df.loc[i, "y"] = tracked_ball_points[i][1] # y
        ball_points_df.loc[i, "d"] = 2 * tracked_ball_points[i][2] # diameter

    ball_points_df = ball_points_df.iloc[2:] # Drop first 2 frames since tracknet requires 3 consecutive frame inputs
    ball_points_df = remove_input_outliers(ball_points_df)
    ball_points_df = ball_points_df.interpolate(method="linear", axis=0) # TODO: Maybe use better method

    ball_points = torch.tensor(ball_points_df.values, dtype=torch.float32)
    intrinsics = torch.tensor(obtain_base_intrinsics(intrinsics_path), dtype=torch.float32)
    extrinsics = torch.tensor(obtain_base_extrinsics(extrinsics_path), dtype=torch.float32)

    batch = create_inference_example(ball_points, intrinsics, extrinsics)
    pred_coords, pred_eot, _, _ = run_model(args.fm, inference_example=batch)

    frame_indices = ball_points_df.index.to_numpy()

    return pred_coords, pred_eot, frame_indices


def save_trajectory_3d(pred_coords: np.ndarray, pred_eot: np.ndarray, frame_indices: np.ndarray, inference_base_folder):
    """
    - pred_coords is (L, 3)
    - pred_eot is (L,)
    - frame_indices is (L,)
    """
    save_path = os.path.join(inference_base_folder, "preds.npz")
    np.savez(save_path, coords=pred_coords, eot=pred_eot, frame_indices=frame_indices)


def create_inference_example(trajectory_2d: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor):
    """
    Inputs are all torch tensors
    - trajectory_2d is (L, 3) [x, y, d]
    - intrinsics is (3, 3)
    - extrinsics is (4, 4)
    """
    L = trajectory_2d.shape[0]
    # Main data
    trajectory = trajectory_2d.unsqueeze(0) # (1, L, 3)
    inputs = trajectory[:, :, :2] # (1, L, 2)
    diameters = torch.stack((trajectory[:, 0, 2], trajectory[:, -1, 2]), axis=1) # (1, 2)
    lengths = torch.tensor([L]) # (1,)
    
    # Other necessary data (unused for real inference)
    coord_labels = torch.zeros((1, L, 3))
    eot_labels = torch.zeros((1, L, 1))
    bounce_token_labels = torch.zeros((1, L, 1))


    batch = (inputs, coord_labels, eot_labels, bounce_token_labels, lengths, intrinsics.unsqueeze(0), extrinsics.unsqueeze(0), diameters)
    return batch


def remove_input_outliers(ball_points_df: pd.DataFrame, threshold_px=100):
    """
    Replaces input coords that are more than threshold_px away from neighbor midpoint w/ NaN
    """
    ball_points_df = ball_points_df.copy()

    while True:
        midpoints = (ball_points_df.shift(1) + ball_points_df.shift(-1)) / 2
        deviations = np.sqrt((ball_points_df["x"] - midpoints["x"])**2 + (ball_points_df["y"] - midpoints["y"])**2)

        if not deviations.max() > threshold_px: # No more outliers
            break

        ball_points_df.iloc[deviations.argmax()] = np.nan

    return ball_points_df


if __name__ == "__main__":
    main()