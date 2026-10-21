from convert_2d_to_3d.paths import Paths
from convert_2d_to_3d.physics import generate_random_trajectory
from convert_2d_to_3d import settings
from convert_2d_to_3d.settings import SyntheticDataGeneration
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def remove_path_before(full_path, first_part_to_keep):
    """
    Takes in a full path. Removes all parent directories up to first_part_to_keep (which is still there after the final cut) to obtain a relative path
    """
    parts = full_path.split(os.sep)

    idx = parts.index(first_part_to_keep)
    return os.sep.join(parts[idx:])


def generate_raw_trajectories(data_gen_settings=SyntheticDataGeneration()):
    """
    - Simulates raw 3D trajectories and writes them to the raw data folder
    - data.csv lists each trajectory's id and relative file location (for use in later processing)
    """

    paths = Paths()
    raw_data_folder = paths.raw_data_folder

    trajectories_3d_folder = os.path.join(raw_data_folder, "trajectories_3d")
    trajectories_eot_folder = os.path.join(raw_data_folder, "trajectories_eot")
    trajectories_bounce_tokens_folder = os.path.join(raw_data_folder, "trajectories_bounce_tokens")

    os.makedirs(trajectories_3d_folder, exist_ok=True)
    os.makedirs(trajectories_eot_folder, exist_ok=True)
    os.makedirs(trajectories_bounce_tokens_folder, exist_ok=True)

    num_trajectories = data_gen_settings.num_camera_configs * data_gen_settings.trajectories_per_config

    rows = []
    for trajectory_id in tqdm(range(num_trajectories), desc="Trajectories"):

        trajectory, times, eot, bounce_tokens = generate_random_trajectory(data_gen_settings)

        trajectory_name = f"trajectory_{trajectory_id:05d}"

        trajectory_3d_path = os.path.join(trajectories_3d_folder, f"{trajectory_name}.npy")
        trajectory_eot_path = os.path.join(trajectories_eot_folder, f"{trajectory_name}.npy")
        trajectory_bounce_tokens_path = os.path.join(trajectories_bounce_tokens_folder, f"{trajectory_name}.npy")

        np.save(trajectory_3d_path, trajectory)
        np.save(trajectory_eot_path, eot)
        np.save(trajectory_bounce_tokens_path, bounce_tokens)

        rows.append({
            "trajectory_id": trajectory_id,
            "simulation_timestep": data_gen_settings.trajectory_generation_settings.simulation_timestep,
            "trajectory_3d_path": remove_path_before(full_path=trajectory_3d_path, first_part_to_keep="trajectories_3d"),
            "trajectory_eot_path": remove_path_before(full_path=trajectory_eot_path, first_part_to_keep="trajectories_eot"),
            "trajectory_bounce_tokens_path": remove_path_before(full_path=trajectory_bounce_tokens_path, first_part_to_keep="trajectories_bounce_tokens")
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(raw_data_folder, "data.csv"), index=False)

    return df


if __name__ == "__main__":
    generate_raw_trajectories()
