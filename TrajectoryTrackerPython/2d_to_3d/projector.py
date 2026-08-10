import cv2
from data_models import CourtProperties
from enum import Enum
import json
from manual_keypoint_labeler import obtain_keypoints
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import os
from overlay import overlay_trajectory_on_image
from projection_component_paths import Paths
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


def obtain_base_intrinsics(filepath):

    with open(filepath, "r") as f:
        data = json.load(f)

    fx = data["fx"]
    fy = data["fy"]
    cx = data["cx"]
    cy = data["cy"]

    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    return intrinsics


def obtain_extrinsics(coord_set: WorldCoordSet, image_points: np.ndarray, intrinsics: np.ndarray, distortion_coeffs=None, verbose=False):
    """
    - intrinsics - a 3 x 3 ndarray
    - image_points - a 4 x 2 ndarray
    - distortional_coeffs - optional 1D ndarray of size 4, 5, 8, 12, or 14
    """
    world_points = _generate_3d_baseline_coords(coord_set)
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


def get_base_camera_config():
    paths = Paths()
    base_intrinsics = obtain_base_intrinsics(paths.base_intrinsics_path)

    keypoint_path = paths.saved_keypoints_path
    if os.path.exists(keypoint_path):
        image_points = np.load(keypoint_path)
    else:
        obtain_keypoints(verbose=True) # Saves to keypoint_path
        image_points = np.load(keypoint_path)

    base_extrinsics = obtain_extrinsics(WorldCoordSet.INNER_BASELINE, image_points, base_intrinsics)
    if base_extrinsics is None:
        raise RuntimeError("Failed to obtain base extrinsics.")

    return base_intrinsics, base_extrinsics


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


def sample_camera_config(rng: np.random.Generator, base_intrinsics: np.ndarray, base_extrinsics: np.ndarray, intrinsics_jitter=5.0, translation_jitter=0.3, rotation_jitter_deg=2.0):
    """
    Perturbs a base camera's intrinsics and extrinsics
    - intrinsics_jitter - max abs pixels of noise added to each of fx, fy, cx, cy
    - translation_jitter - max abs meters of noise added to T
    - rotation_jitter_deg - max abs degrees of noise applied as a small random rotation, composed on top of the base rotation
    """
    intrinsics = base_intrinsics.copy()
    intrinsics[0, 0] += rng.uniform(-intrinsics_jitter, intrinsics_jitter) # fx
    intrinsics[1, 1] += rng.uniform(-intrinsics_jitter, intrinsics_jitter) # fy
    intrinsics[0, 2] += rng.uniform(-intrinsics_jitter, intrinsics_jitter) # cx
    intrinsics[1, 2] += rng.uniform(-intrinsics_jitter, intrinsics_jitter) # cy

    base_r_matrix = base_extrinsics[0:3, 0:3]
    base_t_vec = base_extrinsics[0:3, 3]

    rotation_jitter_rad = np.deg2rad(rotation_jitter_deg)
    random_r_vec = rng.uniform(-rotation_jitter_rad, rotation_jitter_rad, size=3).astype(np.float32)
    random_r_matrix, _ = cv2.Rodrigues(random_r_vec) # Small random rotation matrix

    extrinsics = np.identity(4, dtype=np.float32)
    extrinsics[0:3, 0:3] = random_r_matrix @ base_r_matrix
    extrinsics[0:3, 3] = base_t_vec + rng.uniform(-translation_jitter, translation_jitter, size=3)

    return intrinsics, extrinsics


def project_trajectory_to_2d(trajectory: np.ndarray, projection_matrix: np.ndarray):
    """
    Returns a ndarray of 2d projected trajectory points
    - trajectory is ndarray of 3D points
    """
    projected_trajectory = []
    for point in trajectory:
        homogeneous_point = homogenize(point)
        homogeneous_projection = np.matmul(projection_matrix, homogeneous_point)
        projection = dehomogenize(homogeneous_projection)
        projected_trajectory.append(projection)
    
    return np.stack(projected_trajectory, axis=0)


def _plot_fading_line_2d(ax, x, y, fade_window, label=None, color=(0.0, 0.0, 1.0)):
    """
    The newest segment is fully opaque; opacity ramps down linearly over the trailing fade_window points, and anything older than fade_window points back is omitted entirely
    """
    n_points = len(x)
    start_index = max(0, n_points - 1 - int(round(fade_window)))

    xs = np.asarray(x[start_index:])
    ys = np.asarray(y[start_index:])
    if len(xs) < 2:
        return

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # age (in points) of each segment's newer endpoint; 0 = the current/newest frame
    newest_index = n_points - 1
    segment_end_indices = start_index + np.arange(1, len(segments) + 1)
    ages = newest_index - segment_end_indices
    alphas = np.clip(1.0 - ages / fade_window, 0.0, 1.0) if fade_window > 0 else np.ones(len(segments))

    colors = [(*color, a) for a in alphas]
    ax.add_collection(LineCollection(segments, colors=colors, linewidth=2, label=label))


def plot_2d_trajectory(trajectory, ax=None, fade_window=None):
    """
    If ax is None, creates a new figure/axes and calls plt.show() at the end (standalone use);
    else, draws onto it and leaves creating/showing the figure to the caller (eg for embedding in GUI)
    """
    court_properties = CourtProperties()
    l = court_properties.court_length
    w = court_properties.court_width

    # t = np.linspace(0, end_time, int(10/0.001))
    x = [point[0] for point in trajectory]
    y = [1080 - point[1] for point in trajectory]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    if fade_window is not None and len(x) > 1:
        _plot_fading_line_2d(ax, x, y, fade_window, label="Projected 2D Trajectory")
    else:
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

    if standalone:
        plt.show()



if __name__ == "__main__":
    paths = Paths()

    # intrinsics = np.array([
    #     [1279.2994, 0, 956.8706],
    #     [0, 1279.2994, 550.0753],
    #     [0, 0, 1]
    # ]).astype(np.float32)

    # image_points = np.array([
    #     [759, 603], # Top left
    #     [1112, 604], # Top right
    #     [18, 881], # Bottom left
    #     [1796, 883] # Bottom right
    # ]).astype(np.float32)

    # keypoint_path = paths.saved_keypoints_path
    # if os.path.exists(keypoint_path):
    #     image_points = np.load(keypoint_path)
    # else:
    #     obtain_keypoints(verbose=True)
    #     image_points = np.load(keypoint_path)

    # intrinsics = obtain_base_intrinsics(paths.base_intrinsics_path)
    # extrinsics = obtain_extrinsics(WorldCoordSet.INNER_BASELINE, image_points, intrinsics, verbose=True)

    intrinsics, extrinsics = get_base_camera_config()

    print(extrinsics if extrinsics is not None else "Failure to obtain extrinsics")

    if extrinsics is not None:
        trajectory, _ = generate_sample_trajectory()
        projection_matrix = create_projection_matrix(simplified_intrinsics=intrinsics, extrinsics=extrinsics)
        projected_trajectory = project_trajectory_to_2d(trajectory, projection_matrix)

        overlay_trajectory_on_image(image_path="./sample_img.png", projected_trajectory=projected_trajectory, output_path="./sample_img_overlay.png")

        plot_2d_trajectory(projected_trajectory)


