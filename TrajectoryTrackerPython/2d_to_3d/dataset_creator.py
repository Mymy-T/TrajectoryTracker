import numpy as np
import os
import pandas as pd
from physics import generate_random_trajectory
from paths import Paths
import projector as proj
from settings import SyntheticDataGeneration
from tqdm import tqdm


def _remove_path_before(full_path, first_part_to_keep):
    """
    Takes in a full path. Removes all parent directories up to first_part_to_keep (which is still there after the final cut) to obtain a relative path
    """
    parts = full_path.split(os.sep)
    
    idx = parts.index(first_part_to_keep)
    return os.sep.join(parts[idx:])



def create_dataset(data_gen_settings=SyntheticDataGeneration(), seed=0, timestep=0.001):
    """
    Generates a synthetic dataset of (camera config, 3D trajectory, 2D projected trajectory) instances

    For each of num_camera_configs randomly sampled camera configs (centered around a fixed base config):
        Generate trajectories_per_config random 3D trajectories
        Project to to 2D
        Save both the 3D and 2D trajectory as .npy files
        One row per example  written to csv file
    """

    rng = np.random.default_rng(seed)
    base_intrinsics, base_extrinsics = proj.get_base_camera_config()

    paths = Paths()
    dataset_folder = paths.synthetic_data_folder

    trajectories_3d_folder = os.path.join(dataset_folder, "trajectories_3d")
    trajectories_2d_folder = os.path.join(dataset_folder, "trajectories_2d")
    trajectories_eot_folder = os.path.join(dataset_folder, "trajectories_eot")

    os.makedirs(trajectories_3d_folder, exist_ok=True)
    os.makedirs(trajectories_2d_folder, exist_ok=True)
    os.makedirs(trajectories_eot_folder, exist_ok=True)


    rows = []
    for camera_config_id in tqdm(range(data_gen_settings.num_camera_configs), desc="Camera configs"):

        intrinsics, extrinsics = proj.sample_camera_config(rng, base_intrinsics, base_extrinsics)
        projection_matrix = proj.create_projection_matrix(simplified_intrinsics=intrinsics, extrinsics=extrinsics)

        for trajectory_id in tqdm(range(data_gen_settings.trajectories_per_config), desc=f"Trajectories for camera config {camera_config_id}", leave=False):

            trajectory, times, eot = generate_random_trajectory(rng, timestep=timestep, generation_settings=data_gen_settings.trajectory_generation_settings)
            projected_trajectory = proj.project_trajectory_to_2d(trajectory, projection_matrix)

            example_id = f"camera_config_{camera_config_id:04d}__trajectory_{trajectory_id:04d}"

            trajectory_3d_path = os.path.join(trajectories_3d_folder, f"{example_id}.npy")
            trajectory_2d_path = os.path.join(trajectories_2d_folder, f"{example_id}.npy")
            trajectory_eot_path = os.path.join(trajectories_eot_folder, f"{example_id}.npy")

            np.save(trajectory_3d_path, trajectory)
            np.save(trajectory_2d_path, projected_trajectory)
            np.save(trajectory_eot_path, eot)

            rows.append({
                "camera_config_id": camera_config_id,
                "trajectory_id": trajectory_id,
                "fx": intrinsics[0, 0],
                "fy": intrinsics[1, 1],
                "cx": intrinsics[0, 2],
                "cy": intrinsics[1, 2],
                **{f"extrinsics_{i}": v for i, v in enumerate(extrinsics.flatten())},
                "trajectory_3d_path": _remove_path_before(full_path=trajectory_3d_path, first_part_to_keep="trajectories_3d"),
                "trajectory_2d_path": _remove_path_before(full_path=trajectory_2d_path, first_part_to_keep="trajectories_2d"),
                "trajectory_eot_path": _remove_path_before(full_path=trajectory_eot_path, first_part_to_keep="trajectories_eot"),
                "num_points": len(trajectory),
                "timestep": timestep,
                "bounces": data_gen_settings.trajectory_generation_settings.bounces,
                "hits": data_gen_settings.trajectory_generation_settings.racket_hits
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(dataset_folder, "dataset.csv"), index=False)
    
    return df


if __name__ == "__main__":
    create_dataset()
