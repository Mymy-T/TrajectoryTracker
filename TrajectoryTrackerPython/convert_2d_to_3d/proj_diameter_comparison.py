import argparse
from convert_2d_to_3d.data_models import BallState, CourtProperties, FHInitialDistributionParameters, TrajectoryMode
from convert_2d_to_3d.physics import generate_test_trajectory
from convert_2d_to_3d.projector import obtain_projected_diameters
from convert_2d_to_3d.settings import SyntheticDataGeneration
from convert_2d_to_3d.visualizer import plot_3d_trajectory
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# These 4 constants are for 2 different view types and were determined by directly accessing the saved intrinsics and extrinsics for the broadcast and eye-level camera settings
BROADCAST_INTRINSICS = np.array([[2860.131822980122, 0, 960.0],
                                    [0, 2860.131822980122, 540.0],
                                    [0, 0, 1]])
BROADCAST_EXTRINSICS = np.array([[-1.73734997e-02, -4.57712424e-03,  9.99838593e-01, -5.84468124e-01],
                                    [-3.12027909e-01, -9.50022690e-01, -9.77096579e-03, -7.96077530e-01],
                                    [9.49914073e-01, -3.12147302e-01,  1.50770285e-02, 3.37050828e+01],
                                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

EYE_LEVEL_INTRINSICS = np.array([[1279.2994, 0, 956.8706],
                                    [0, 1279.2994, 550.0753],
                                    [0, 0, 1]])
EYE_LEVEL_EXTRINSICS = np.array([[-1.16046555e-02, 1.23608517e-03, 9.99931900e-01, -3.55336549e-01],
                                    [-1.29937221e-02, -9.99914989e-01,  1.08526633e-03, 1.38137324e+00],
                                    [ 9.99848236e-01, -1.29802431e-02,  1.16197303e-02, 1.78139154e+01],
                                    [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

CONSTANT_Y = 1 # meters

LOWER_Y_LIM = 1 # Empirical, set after seeing graph
UPPER_Y_LIM = 15

def main():
    parser = argparse.ArgumentParser(description="A program to show how projected diameter changes throughout a trajectory at different camera settings.")
    parser.add_argument("--trajectory", action="store_true", help="View the test trajectory. Other args will be ignored if this is passed in.")
    parser.add_argument("--view", choices=["broadcast", "eye-level"], help="The view type in which to plot the diameters over court position")
    args = parser.parse_args()

    trajectory = generate_test_trajectory()
    const_y_trajectory = generate_constant_y_trajectory(CONSTANT_Y)

    if args.trajectory:
        plot_3d_trajectory(trajectory)
        plot_3d_trajectory(const_y_trajectory, label="Constant Y Trajectory") # Sequentially presented, x out of first to see second
    else:
        if args.view == "broadcast":
            intrinsics = BROADCAST_INTRINSICS
            extrinsics = BROADCAST_EXTRINSICS
        elif args.view == "eye-level":
            intrinsics = EYE_LEVEL_INTRINSICS
            extrinsics = EYE_LEVEL_EXTRINSICS
        else:
            raise RuntimeError("--view must be passed if --trajectory isn't.")
        
        diameters = obtain_projected_diameters(trajectory, intrinsics, extrinsics, all_diameters=True)
        const_y_diameters = obtain_projected_diameters(const_y_trajectory, intrinsics, extrinsics, all_diameters=True)

        diameters_min = np.min(diameters)
        diameters_max = np.max(diameters)
        diameters_range = np.max(diameters) - np.min(diameters)
        const_y_diameters_min = np.min(const_y_diameters)
        const_y_diameters_max = np.max(const_y_diameters)
        const_y_diameters_range = np.max(const_y_diameters) - np.min(const_y_diameters)

        headers = ["Type", "Min", "Max", "Range"]
        data = [["Test Trajectory", diameters_min, diameters_max, diameters_range],
                  [f"Constant Y Trajectory (Y = {CONSTANT_Y})", const_y_diameters_min, const_y_diameters_max, const_y_diameters_range]]

        print(tabulate(data, headers=headers, tablefmt="grid"))

        x = trajectory[:, 0]
        x2 = const_y_trajectory[:, 0]

        plot(diameters, x, const_y_diameters, x2, CONSTANT_Y)



def generate_constant_y_trajectory(y):
    l = CourtProperties().court_length
    fh_dist = FHInitialDistributionParameters()
    vx = fh_dist.v_mu[0]
    x = -l/2
    dt = SyntheticDataGeneration().trajectory_generation_settings.simulation_timestep

    trajectory = []
    
    while x < l/2:
        trajectory.append([x, y, 0])
        x += vx * dt
    
    return np.array(trajectory).astype(np.float64) # (L, 3)



def plot(diameters: np.ndarray, x: np.ndarray, const_y_diameters: np.ndarray, x2: np.ndarray, const_y: int):

    fig, (ax, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    ax.scatter(x, diameters)
    ax.set_xlabel("Court X Position (m)")
    ax.set_ylabel("Projected Diameter (px)")
    ax.set_title("Real Trajectory")
    ax.set_ylim(LOWER_Y_LIM, UPPER_Y_LIM)

    ax2.scatter(x2, const_y_diameters)
    ax2.set_xlabel("Court X Position (m)")
    ax2.set_ylabel("Projected Diameter (px)")
    ax2.set_title(f"Constant Y Trajectory (Y = {const_y}m)")
    ax2.set_ylim(LOWER_Y_LIM, UPPER_Y_LIM)

    plt.show()



if __name__ == "__main__":
    main()