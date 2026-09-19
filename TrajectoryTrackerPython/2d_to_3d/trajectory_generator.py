import numpy as np
import os
import pandas as pd
from paths import Paths
from physics import generate_random_trajectory
import settings
from settings import SyntheticDataGeneration
from tqdm import tqdm


def remove_path_before(full_path, first_part_to_keep):
    """
    Takes in a full path. Removes all parent directories up to first_part_to_keep (which is still there after the final cut) to obtain a relative path
    """
    parts = full_path.split(os.sep)

    idx = parts.index(first_part_to_keep)
    return os.sep.join(parts[idx:])


def generate_raw_trajectories(data_gen_settings=SyntheticDataGeneration(), seed=0, timestep=settings.simulation_timestep):
    """
    - Simulates raw 3D trajectories and writes them to the raw data folder
    - data.csv lists each trajectory's id and relative file location (for use in later processing)
    """

    rng = np.random.default_rng(seed)

    paths = Paths()
    raw_data_folder = paths.raw_data_folder

    trajectories_3d_folder = os.path.join(raw_data_folder, "trajectories_3d")
    trajectories_eot_folder = os.path.join(raw_data_folder, "trajectories_eot")

    os.makedirs(trajectories_3d_folder, exist_ok=True)
    os.makedirs(trajectories_eot_folder, exist_ok=True)

    num_trajectories = data_gen_settings.num_camera_configs * data_gen_settings.trajectories_per_config

    rows = []
    for trajectory_id in tqdm(range(num_trajectories), desc="Trajectories"):

        trajectory, times, eot = generate_random_trajectory(rng, timestep=timestep, generation_settings=data_gen_settings.trajectory_generation_settings)

        trajectory_name = f"trajectory_{trajectory_id:05d}"

        trajectory_3d_path = os.path.join(trajectories_3d_folder, f"{trajectory_name}.npy")
        trajectory_eot_path = os.path.join(trajectories_eot_folder, f"{trajectory_name}.npy")

        np.save(trajectory_3d_path, trajectory)
        np.save(trajectory_eot_path, eot)

        rows.append({
            "trajectory_id": trajectory_id,
            "simulation_timestep": timestep,
            "trajectory_3d_path": remove_path_before(full_path=trajectory_3d_path, first_part_to_keep="trajectories_3d"),
            "trajectory_eot_path": remove_path_before(full_path=trajectory_eot_path, first_part_to_keep="trajectories_eot")
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(raw_data_folder, "data.csv"), index=False)

    return df


if __name__ == "__main__":
    generate_raw_trajectories()
