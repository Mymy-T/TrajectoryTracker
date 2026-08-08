import cv2
from data_models import CourtProperties
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
from overlay import overlay_trajectory_on_image
from physics import generate_sample_trajectory

class WorldCoordSet(Enum):
    SERVICE_LINE = 1
    INNER_BASELINE = 2 


def _generate_3d_baseline_coords(coord_set: WorldCoordSet):
    """
    Returns a ndarray of size 4 x 3; each row is a 3D coord; the rows are ordered as far_left, far_right, near_left, near_right, in that precise order
    """
    court_properties = CourtProperties()
    w = court_properties.court_width
    l = court_properties.court_length
    sw = court_properties.singles_width
    sld = court_properties.service_line_dist_from_net
    if coord_set == WorldCoordSet.SERVICE_LINE:
        far_left = np.array([sld, 0, -sw/2]).astype(np.float32)
        far_right = np.array([sld, 0, sw/2]).astype(np.float32)
        near_left = np.array([-sld, 0, -sw/2]).astype(np.float32)
        near_right = np.array([-sld, 0, sw/2]).astype(np.float32)
    elif coord_set == WorldCoordSet.INNER_BASELINE:
        far_left = np.array([l/2, 0, -sw/2]).astype(np.float32)
        far_right = np.array([l/2, 0, sw/2]).astype(np.float32)
        near_left = np.array([-l/2, 0, -sw/2]).astype(np.float32)
        near_right = np.array([-l/2, 0, sw/2]).astype(np.float32)
    
    return np.stack([far_left, far_right, near_left, near_right], axis=0)


def _expand_intrinsics_matrix(intrinsics: np.ndarray):
    """
    Expands a 3 x 3 intrinsics matrix to the full 3 x 4 form
    """
    zero_vec = np.zeros((3, 1))
    return np.concatenate((intrinsics, zero_vec), axis=1)


def obtain_extrinsics(world_points: np.ndarray, image_points: np.ndarray, intrinsics: np.ndarray, distortion_coeffs=None, verbose=False):
    """
    - intrinsics - a 3 x 3 ndarray
    - image_points - a 4 x 2 ndarray
    - world_points - a 4 x 3 ndarray
    - distortional_coeffs - optional 1D ndarray of size 4, 5, 8, 12, or 14
    """
    success, r_vec, t_vec = cv2.solvePnP(world_points, image_points, intrinsics, distortion_coeffs, flags=cv2.SOLVEPNP_IPPE)
    if success:
        r_matrix, _ = cv2.Rodrigues(r_vec) # Convert rotation vector into 3 x 3 rotation matrix

        if verbose:
            print("Each column of the rotation matrix is a basis vector of the camera coordinate system in terms of world coordinates. The first column is +x (right in camera view), second column is +y (down in camera view), and third column is +z (straight forward in camera view)")
            print(f"Rotation matrix: \n{r_matrix}")
            print(f"Camera position (in world coords): \n{-np.matmul(np.transpose(r_matrix), t_vec)}")

        # t_vec is the position of the world origin from camera POV; therefore, t_vec is in camera frame and t_vec = R(P_w - t) = R(0 - t) = -Rt where R is 3 x 3 and t is 3 x 1 (simplified or heterogeneous form)
        extrinsics = np.identity(4, dtype=np.float32)
        extrinsics[0:3, 0:3] = r_matrix
        extrinsics[0:3, 3] = t_vec.flatten() # Last col of extrinsic matrix is -Rt = t_vec
        return extrinsics
    
    return None


def homogenize(coords: np.ndarray):
    if coords.ndim == 1:
        return np.append(coords, 1.0)
        
    ones = np.ones((coords.shape[0], 1), dtype=coords.dtype)
    return np.concatenate((coords, ones), axis=1)


def dehomogenize(coords: np.ndarray):
    if coords.ndim == 1:
        return coords[:-1] / coords[-1]
    
    return coords[:, :-1] / coords[:, -1:]


def create_projection_matrix(simplified_intrinsics: np.ndarray, extrinsics: np.ndarray):
    return np.matmul(_expand_intrinsics_matrix(simplified_intrinsics), extrinsics) # https://visionbook.mit.edu/imaging_geometry.html#full-camera-model


def project_trajectory_to_2d(trajectory: list, projection_matrix: np.ndarray):
    """
    Returns a list of 2d projected trajectory points
    - trajectory is a list of 1D length 3 ndarrays (position vectors)
    """
    projected_trajectory = []
    for point in trajectory:
        homogeneous_point = homogenize(point)
        homogeneous_projection = np.matmul(projection_matrix, homogeneous_point)
        projection = dehomogenize(homogeneous_projection)
        projected_trajectory.append(projection)
    
    return projected_trajectory


def plot_2d_trajectory(trajectory):
    court_properties = CourtProperties()
    l = court_properties.court_length
    w = court_properties.court_width

    # t = np.linspace(0, end_time, int(10/0.001))
    x = [point[0] for point in trajectory]
    y = [1080 - point[1] for point in trajectory]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, label="Projected 2D Trajectory", color="b", linewidth=2)

    # Highlight start and end points
    ax.scatter(x[0], y[0], color="green", s=50, label="Start")
    ax.scatter(x[-1], y[-1], color="red", s=50, label="End")

    # Set limits based on court dimensions or pixel limits if applicable
    # TODO: convert to 640 by 360 to simulate true data
    ax.set_xlim([0, 1920])
    ax.set_ylim([0, 1080])

    ax.set_xlabel("X Axis (2D Projection)")
    ax.set_ylabel("Y Axis (2D Projection)")
    ax.set_title("2D Projected Trajectory")
    ax.legend()

    plt.show()



if __name__ == "__main__":
    intrinsics = np.array([
        [1279.2994, 0, 956.8706],
        [0, 1279.2994, 550.0753],
        [0, 0, 1]
    ]).astype(np.float32)
    image_points = np.array([
        [759, 603], # Top left
        [1112, 604], # Top right
        [18, 881], # Bottom left
        [1796, 883] # Bottom right
    ]).astype(np.float32)

    world_points = _generate_3d_baseline_coords(WorldCoordSet.INNER_BASELINE)

    extrinsics = obtain_extrinsics(world_points, image_points, intrinsics, verbose=True)

    print(extrinsics if extrinsics is not None else "Failure to obtain extrinsics")

    if extrinsics is not None:
        trajectory, end_time = generate_sample_trajectory()
        projection_matrix = create_projection_matrix(simplified_intrinsics=intrinsics, extrinsics=extrinsics)
        projected_trajectory = project_trajectory_to_2d(trajectory, projection_matrix)

        overlay_trajectory_on_image(image_path="./sample_img.png", projected_trajectory=projected_trajectory, output_path="./sample_img_overlay.png")

        plot_2d_trajectory(projected_trajectory)


