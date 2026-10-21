from convert_2d_to_3d.physics import generate_sample_trajectory
from fractions import Fraction
import numpy as np

def _format_timestep(timestep: float):
    return str(Fraction(timestep).limit_denominator(10000))


def _format_point(point: np.ndarray):
    return f"({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})"


def assess_final_position(timestep_options: list):

    timestep_width = max(len(_format_timestep(timestep)) for timestep in timestep_options)
    print("All positions are formatted as (x, y, z)")

    for timestep in timestep_options:
        trajectory, _, _ = generate_sample_trajectory(timestep, bounces=True)
        no_bounce_trajectory, _, _ = generate_sample_trajectory(timestep, bounces=False)

        final_position = trajectory[-1]
        no_bounce_final_position = no_bounce_trajectory[-1]

        print(f"Timestep: {_format_timestep(timestep).ljust(timestep_width)} s\t Final position: {_format_point(final_position)} m\t No-bounce final position: {_format_point(no_bounce_final_position)} m")



def assess_trajectory_error(timesteps: list, reference_timestep: float):

    if min(timesteps) < reference_timestep:
        print("ERROR: reference_timestep must be smaller than the smallest timestep in timesteps.")
        return


    reference_trajectory, reference_times, _ = generate_sample_trajectory(reference_timestep, bounces=True)

    no_bounce_reference_trajectory, no_bounce_reference_times, _ = generate_sample_trajectory(reference_timestep, bounces=False)

    print(f"\nReference timestep: {_format_timestep(reference_timestep)} s")
    print("-" * 20)

    timestep_width = max(len(_format_timestep(timestep)) for timestep in timesteps)

    for timestep in timesteps:

        # Bounce trajectories
        trajectory, times, _ = generate_sample_trajectory(timestep, bounces=True)
        no_bounce_trajectory, no_bounce_times, _ = generate_sample_trajectory(timestep, bounces=False)

        valid_mask = times <= reference_times[-1]
        times_valid = times[valid_mask]
        trajectory_valid = trajectory[valid_mask]

        reference_interp = np.column_stack( [np.interp(times_valid, reference_times, reference_trajectory[:, dim]) for dim in range(3)] )

        errors = np.linalg.norm(trajectory_valid - reference_interp, axis=1)

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        max_error = np.max(errors)

        # No bounce trajectories
        no_bounce_valid_mask = no_bounce_times <= no_bounce_reference_times[-1]
        no_bounce_times_valid = no_bounce_times[no_bounce_valid_mask]
        no_bounce_trajectory_valid = no_bounce_trajectory[no_bounce_valid_mask]

        no_bounce_reference_interp = np.column_stack( [np.interp(no_bounce_times_valid, no_bounce_reference_times, no_bounce_reference_trajectory[:, dim]) for dim in range(3)] )

        no_bounce_errors = np.linalg.norm(no_bounce_trajectory_valid - no_bounce_reference_interp, axis=1)

        no_bounce_mae = np.mean(no_bounce_errors)
        no_bounce_rmse = np.sqrt(np.mean(no_bounce_errors ** 2))
        no_bounce_max_error = np.max(no_bounce_errors)


        print(f"Timestep: {_format_timestep(timestep).ljust(timestep_width)} s\t MAE: {mae:.6f} m\t RMSE: {rmse:.6f} m\t Max abs error: {max_error:.6f} m\t No bounce MAE: {no_bounce_mae:.6f} m\t No bounce RMSE: {no_bounce_rmse:.6f} m\t No bounce max abs error: {no_bounce_max_error:.6f} m")

if __name__ == "__main__":
    timesteps = sorted([1.0/6, 1.0/30, 1.0/60, 1.0/120, 1.0/600, 1.0/1200, 1.0/3000, 1.0/6000], reverse=True)

    command_recognized = False
    while not command_recognized:
        check_final_positions = input("Assess the final positions for varying timstep choices? (y/n): ")
        if check_final_positions.lower() == "y":
            command_recognized = True
            assess_final_position(timesteps)
        elif check_final_positions.lower() == "n":
            command_recognized = True

    command_recognized = False
    while not command_recognized:
        check_trajectory_errors = input(f"Aassess overall trajectory error for varying timestep choices, with {_format_timestep(timesteps[-1])} s timestep as a reference? (y/n): ")
        if check_trajectory_errors.lower() == "y":
            command_recognized = True
            assess_trajectory_error(timesteps[:-1], reference_timestep=timesteps[-1])
        elif check_trajectory_errors.lower() == "n":
            command_recognized = True