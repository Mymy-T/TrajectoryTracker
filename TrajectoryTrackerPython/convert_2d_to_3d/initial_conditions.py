import argparse
from convert_2d_to_3d.data_models import BallState
from convert_2d_to_3d.paths import Paths
from convert_2d_to_3d.physics import PhysicsSimulation, generate_test_trajectory, get_test_trajectory_spin, get_test_trajectory_velocity, interpolate_to_new_timestep
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
    parser.add_argument("--labels", action="store_true", help="Run the fit on the trajectory coord labels for the prediction; then the initial condition parameters can be compared across prediction and label")
    parser.add_argument("--test", action="store_true", help="Run the fit on a test trajectory with known initial conditions.")
    args = parser.parse_args()

    if args.labels and args.test:
        raise RuntimeError("--labels and --test are mutually exclusive.")

    simulation_dt = SyntheticDataGeneration().trajectory_generation_settings.simulation_timestep
    time_per_frame = 1 / SyntheticDataGeneration().target_fps

    if not args.test:
        trajectory_path = Paths().inference_trajectory_path
        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(f"{trajectory_path} doesn't exist yet, run inference first.")

        data = np.load(trajectory_path)

        if not args.labels:
            sub_trajectories = split_trajectory(data["coords"])
        else:
            sub_trajectories = split_trajectory(data["coord_labels"])
    else:
        physics_trajectory = generate_test_trajectory()
        interoplated_trajectory, _, _ = interpolate_to_new_timestep(physics_trajectory, eot=None, bounce_tokens=None, dt_0=simulation_dt, dt_1=time_per_frame)
        sub_trajectories = [interoplated_trajectory]

    trajectory = sub_trajectories[0] # TODO: Loop over all
    L = len(trajectory)

    times = np.arange(L) * time_per_frame
    measured_positions = trajectory
    pos_0 = trajectory[0, :] # pos_0 and guess auto converted to lists inside objective_func
    guess = [23.0, 5.0, 0.0, -185.0] # [vx, vy, vz, wz] TODO: wx estimation via bounce & constrained wy estimation

    result = minimize(objective_func, guess, args=(pos_0, times, measured_positions), method="Powell")

    if result.success:
        v_0 = result.x[:3]
        w_0 = np.array([0.0, 0.0 , result.x[-1]])

        print(f"Initial velocity: {v_0}")
        print(f"Initial spin: {w_0}")
        print(f"Final fit RMSE (cm): {result.fun * 100}")

        if args.test:
            print(f"Actual initial velocity: {get_test_trajectory_velocity()}")
            print(f"Actual initial spin: {get_test_trajectory_spin()}")

        if args.plot:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), subplot_kw={"projection": "3d"})

            if args.test:
                fig.suptitle("Test Trajectory", fontsize=16)
            elif args.labels:
                fig.suptitle("Synthetic Test Example - Labels", fontsize=16)
            else:
                fig.suptitle("Synthetic Test Example - Predictions", fontsize=16)

            estimated_positions = simulate_trajectory(result.x, pos_0, times)

            plot_3d_trajectory(estimated_positions, ax=ax1, color="b", label="Estimated")
            
            plot_3d_trajectory(trajectory, ax=ax2, color="orange", label="Observed")

            plot_3d_trajectory(trajectory, ax=ax3, color="orange", label="Overlay", hide_legend=True)
            plot_3d_trajectory(estimated_positions, ax=ax3, color="b", label="Overlay", hide_legend=True)

            v_0_str = np.array2string(v_0, formatter={"float_kind": lambda x: f"{x:.2f}"})
            w_0_str = np.array2string(w_0, formatter={"float_kind": lambda x: f"{x:.2f}"})
            initial_conditions_text = f"Initial velocity: {v_0_str}     |     Initial spin: {w_0_str}     |     Final fit RMSE (cm): {result.fun * 100:.2f}"

            fig.text(0.5, 0.02, initial_conditions_text, ha='center', fontsize=10)

            if args.test:
                actual_v_0_str = np.array2string(get_test_trajectory_velocity(), formatter={"float_kind": lambda x: f"{x:.2f}"})
                actual_w_0_str = np.array2string(get_test_trajectory_spin(), formatter={"float_kind": lambda x: f"{x:.2f}"})
                actual_initial_conditions_text = f"Actual initial velocity: {actual_v_0_str}     |     Actual initial spin: {actual_w_0_str}"

                fig.text(0.5, 0.04, actual_initial_conditions_text, ha='center', fontsize=10)

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

    initial_state = list(pos_0) + list(params[:-1]) + [0.0, 0.0, params[-1]]

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