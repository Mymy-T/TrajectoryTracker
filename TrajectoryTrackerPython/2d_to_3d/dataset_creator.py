from multiprocessing import process
import numpy as np
import os
import pandas as pd
from paths import Paths
from physics import interpolate_to_new_timestep
import projector as proj
import settings
from settings import SyntheticDataGeneration
from tqdm import tqdm
from trajectory_generator import remove_path_before


def create_dataset(data_gen_settings=SyntheticDataGeneration(), seed=0, target_fps=settings.target_fps):
    """
    - Builds the training dataset from the raw trajectories in the raw data folder

    - For each raw trajectory:
        Interpolate the 3D trajectory and EOT flags from the simulation timestep down to target_fps
        Project the resampled trajectory to 2D through a sampled camera config
        Compute the apparent ball diameters at the first and last frame
        Save all four arrays as .npy files, one row per example written to dataset.csv

    - Camera configs are assigned in blocks of trajectories_per_config
    - Interpolaion from simulation timestep to target_fps timestep is done before projection/diameter generation
    """

    rng = np.random.default_rng(seed)
    base_intrinsics, base_extrinsics = proj.get_base_camera_config()

    paths = Paths()
    raw_data_folder = paths.raw_data_folder
    processed_data_folder = paths.processed_data_folder

    raw_df = pd.read_csv(os.path.join(raw_data_folder, "data.csv"))

    trajectories_3d_folder = os.path.join(processed_data_folder, "trajectories_3d")
    trajectories_2d_folder = os.path.join(processed_data_folder, "trajectories_2d")
    trajectories_eot_folder = os.path.join(processed_data_folder, "trajectories_eot")
    trajectories_bounce_tokens_folder = os.path.join(processed_data_folder, "trajectories_bounce_tokens")
    trajectories_diameter_folder = os.path.join(processed_data_folder, "trajectories_diameter")

    os.makedirs(trajectories_3d_folder, exist_ok=True)
    os.makedirs(trajectories_2d_folder, exist_ok=True)
    os.makedirs(trajectories_eot_folder, exist_ok=True)
    os.makedirs(trajectories_bounce_tokens_folder, exist_ok=True)
    os.makedirs(trajectories_diameter_folder, exist_ok=True)

    new_timestep = 1 / target_fps

    cam_config_settings = data_gen_settings.camera_config_sampling_settings

    rows = []
    for camera_config_id in tqdm(range(data_gen_settings.num_camera_configs), desc="Camera configs"):

        intrinsics, extrinsics = proj.sample_camera_config(rng, base_intrinsics, base_extrinsics, sampling_settings=cam_config_settings)
        projection_matrix = proj.create_projection_matrix(simplified_intrinsics=intrinsics, extrinsics=extrinsics)

        for trajectory_id in tqdm(range(data_gen_settings.trajectories_per_config), desc=f"Trajectories for camera config {camera_config_id}", leave=False):

            raw_row = raw_df.iloc[camera_config_id * data_gen_settings.trajectories_per_config + trajectory_id, :]

            raw_trajectory = np.load(os.path.join(raw_data_folder, raw_row.loc["trajectory_3d_path"]))
            raw_eot = np.load(os.path.join(raw_data_folder, raw_row.loc["trajectory_eot_path"]))
            raw_bounce_tokens = np.load(os.path.join(raw_data_folder, raw_row.loc["trajectory_bounce_tokens_path"]))

            trajectory, eot, bounce_tokens = interpolate_to_new_timestep(raw_trajectory, raw_eot, raw_bounce_tokens, settings.simulation_timestep, new_timestep)
            projected_trajectory = proj.project_trajectory_to_2d(trajectory, projection_matrix)
            diameters = proj.obtain_projected_diameters(trajectory, intrinsics, extrinsics)

            example_id = f"camera_config_{camera_config_id:04d}__trajectory_{trajectory_id:04d}"

            trajectory_3d_path = os.path.join(trajectories_3d_folder, f"{example_id}.npy")
            trajectory_2d_path = os.path.join(trajectories_2d_folder, f"{example_id}.npy")
            trajectory_eot_path = os.path.join(trajectories_eot_folder, f"{example_id}.npy")
            trajectory_bounce_tokens_path = os.path.join(trajectories_bounce_tokens_folder, f"{example_id}.npy")
            trajectory_diameter_path = os.path.join(trajectories_diameter_folder, f"{example_id}.npy")

            np.save(trajectory_3d_path, trajectory)
            np.save(trajectory_2d_path, projected_trajectory)
            np.save(trajectory_eot_path, eot)
            np.save(trajectory_bounce_tokens_path, bounce_tokens)
            np.save(trajectory_diameter_path, diameters)

            rows.append({
                "camera_config_id": camera_config_id,
                "trajectory_id": trajectory_id,
                "raw_trajectory_id": raw_row.loc["trajectory_id"],
                "fx": intrinsics[0, 0],
                "fy": intrinsics[1, 1],
                "cx": intrinsics[0, 2],
                "cy": intrinsics[1, 2],
                **{f"extrinsics_{i}": v for i, v in enumerate(extrinsics.flatten())},
                "trajectory_3d_path": remove_path_before(full_path=trajectory_3d_path, first_part_to_keep="trajectories_3d"),
                "trajectory_2d_path": remove_path_before(full_path=trajectory_2d_path, first_part_to_keep="trajectories_2d"),
                "trajectory_eot_path": remove_path_before(full_path=trajectory_eot_path, first_part_to_keep="trajectories_eot"),
                "trajectory_bounce_tokens_path": remove_path_before(full_path=trajectory_bounce_tokens_path, first_part_to_keep="trajectories_bounce_tokens"),
                "trajectory_diameter_path": remove_path_before(full_path=trajectory_diameter_path, first_part_to_keep="trajectories_diameter"),
                "num_points": len(trajectory),
                "timestep": new_timestep,
                "bounces": data_gen_settings.trajectory_generation_settings.bounces,
                "hits": data_gen_settings.trajectory_generation_settings.racket_hits
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(processed_data_folder, "dataset.csv"), index=False)

    return df


if __name__ == "__main__":
    create_dataset()
