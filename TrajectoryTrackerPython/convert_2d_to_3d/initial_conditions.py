import argparse
from convert_2d_to_3d.data_models import BallState
from convert_2d_to_3d.paths import Paths
from convert_2d_to_3d.physics import PhysicsSimulation
from convert_2d_to_3d.settings import SyntheticDataGeneration
from convert_2d_to_3d.visualizer import plot_3d_trajectory
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def main():

    parser = argparse.ArgumentParser(description="Program to estimate initial flight conditions from reconstructed trajectory")
    parser.add_argument("--plot", action="store_true", help="Plot observed and trajectory using estimated initial conditions")
    args = parser.parse_args()

    time_per_frame = 1 / SyntheticDataGeneration().target_fps

    trajectory_path = Paths().inference_trajectory_path
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(f"{trajectory_path} doesn't exist yet, run inference first.")

    data = np.load(trajectory_path)
    sub_trajectories = split_trajectory(data["coords"])
    trajectory = sub_trajectories[0] # TODO: Loop over all
    L = len(trajectory)

    times = np.arange(L) * time_per_frame
    measured_positions = trajectory
    pos_0 = trajectory[0, :] # pos_0 and guess auto converted to lists inside objective_func
    guess = [23.0, 5.0, 0.0, 0.0, 0.0, -185.0] # [vx, vy, vz, wx, wy, wz]

    result = minimize(objective_func, guess, args=(pos_0, times, measured_positions), method="Powell")

    if result.success:
        v_0 = result.x[:3]
        w_0 = result.x[3:]

        print(f"Initial velocity: {v_0}")
        print(f"Initial spin: {w_0}")
        print(f"Final error: {result.fun}")

        if args.plot:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"})

            estimated_positions = simulate_trajectory(result.x, pos_0, times)
            plot_3d_trajectory(estimated_positions, ax=ax1, label="Estimated")
            
            plot_3d_trajectory(trajectory, ax=ax2, label="Observed")

            plt.show()

    else:
        print("Fit unsuccessful.")
    

def ode_system(t, state):
    x, y, z, vx, vy, vz, wx, wy, wz = state
    
    ball_state = BallState(position=np.zeros(3), velocity=np.zeros(3), spin=np.zeros(3)) # Unused, just for init to get compute_accelerations func
    simulator = PhysicsSimulation(ball_state, SyntheticDataGeneration())
    a, alpha = simulator.compute_accelerations(v=np.array([vx, vy, vz]), w=np.array([wx, wy, wz]))

    return [vx, vy, vz, a[0], a[1], a[2], alpha[0], alpha[1], alpha[2]]


def objective_func(guess, pos_0, times, measured_positions):

    positions = simulate_trajectory(guess, pos_0, times)

    sq_dists = np.sum((positions - measured_positions)**2, axis=1)
    rmse = np.sqrt(np.mean(sq_dists))

    return rmse


def simulate_trajectory(params, pos_0, times):

    initial_state = list(pos_0) + list(params)

    solution = solve_ivp(ode_system, (times[0], times[-1]), initial_state, t_eval=times)
    positions = solution.y[:3, :].T # (L, 3)

    return positions


def split_trajectory(trajectory: np.ndarray, bounce_vy_bound=15):
    """
    Splits by bounce (dy changes from - to +) and racket hit (dx changes signs)
    - trajectory is (N, 3)
    - bounce_vy_bound is the expected maximum vy a ball can have before/after bounce; can be heuristically determined using rough kinematics estimates
    Returns list of (N, 3) ndarrays, each element corresponding to part of trajectory
    """
    sub_trajectories = []
    start_i = 0
    prev_delta = trajectory[1, :] - trajectory[0, :] # Initially prev_delta = delta so nothing happens
    for i in range(1, len(trajectory)):

        delta = trajectory[i, :] - trajectory[i - 1, :] # (3,)

        dy_0 = prev_delta[1]
        dy = delta[1]
        y_0 = trajectory[i - 1, 1]
        y = trajectory[i, 1]
        dx_0 = prev_delta[0]
        dx = delta[0]
        dt = 1 / SyntheticDataGeneration().target_fps

        bounce_height_threshold = bounce_vy_bound * dt * 1/2 # Factor of 1/2 since the displacement would be split across the bounce in the case that y_0 = y and min(y_0, y) is at maximum
        if dy_0 * dy <= 0 and min(y_0, y) < bounce_height_threshold: # Bounce
            sub_trajectories.append(trajectory[start_i:i, :])
            start_i = i
        elif dx_0 * dx <= 0: # Racket hit
            sub_trajectories.append(trajectory[start_i:i, :])
            start_i = i
        
        prev_delta = delta.copy()

    sub_trajectories.append(trajectory[start_i:, :])

    return sub_trajectories


if __name__ == "__main__":
    main()