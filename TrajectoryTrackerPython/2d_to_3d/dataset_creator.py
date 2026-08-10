import numpy as np
import os
import pandas as pd
from physics import generate_random_trajectory
from projection_component_paths import Paths
import projector as proj


def create_dataset(num_camera_configs=5, trajectories_per_config=10, seed=0, timestep=0.01, bounces=True, hits=True):
    """
    Generates a synthetic dataset of (camera config, 3D trajectory, 2D projected trajectory) instances.

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

    os.makedirs(trajectories_3d_folder, exist_ok=True)
    os.makedirs(trajectories_2d_folder, exist_ok=True)


    rows = []
    for camera_config_id in range(num_camera_configs):

        intrinsics, extrinsics = proj.sample_camera_config(rng, base_intrinsics, base_extrinsics)
        projection_matrix = proj.create_projection_matrix(simplified_intrinsics=intrinsics, extrinsics=extrinsics)

        for trajectory_id in range(trajectories_per_config):

            trajectory, times = generate_random_trajectory(rng, timestep=timestep, bounces=bounces, racket_hits=hits)
            projected_trajectory = proj.project_trajectory_to_2d(trajectory, projection_matrix)

            example_id = f"camera_config_{camera_config_id:04d}__trajectory_{trajectory_id:04d}"

            trajectory_3d_path = os.path.join(trajectories_3d_folder, f"{example_id}.npy")
            trajectory_2d_path = os.path.join(trajectories_2d_folder, f"{example_id}.npy")

            np.save(trajectory_3d_path, trajectory)
            np.save(trajectory_2d_path, projected_trajectory)

            rows.append({
                "camera_config_id": camera_config_id,
                "trajectory_id": trajectory_id,
                "fx": intrinsics[0, 0],
                "fy": intrinsics[1, 1],
                "cx": intrinsics[0, 2],
                "cy": intrinsics[1, 2],
                **{f"extrinsics_{i}": v for i, v in enumerate(extrinsics.flatten())},
                "trajectory_3d_path": trajectory_3d_path,
                "trajectory_2d_path": trajectory_2d_path,
                "num_points": len(trajectory),
                "timestep": timestep,
                "bounces": bounces,
                "hits": hits
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(dataset_folder, "dataset.csv"), index=False)
    
    return df


if __name__ == "__main__":
    create_dataset()
