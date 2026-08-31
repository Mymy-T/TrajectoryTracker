import cv2
from data_models import CourtProperties, CameraConfigSamplingSettings
from enum import Enum
import json
from manual_keypoint_labeler import obtain_keypoints
import numpy as np
import os
from paths import Paths
from physics import generate_sample_trajectory
import torch
from visualizer import overlay_trajectory_on_image, plot_2d_trajectory

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

    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

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
        extrinsics = np.identity(4, dtype=np.float64)
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


def dehomogenize(coords):
    """
    coords is either a numpy ndarray or a torch tensor
    """

    is_numpy = isinstance(coords, np.ndarray)

    if is_numpy:
        tensor_coords = torch.from_numpy(coords)
    else:
        tensor_coords = coords

    if tensor_coords.ndim == 1:
        end_is_zero = torch.isclose(tensor_coords[-1], torch.tensor(0.0, dtype=tensor_coords.dtype, device=tensor_coords.device), atol=1e-7).item()

        final_tensor = tensor_coords[:-1] / tensor_coords[-1] if not end_is_zero else tensor_coords[:-1] # Treat as direction if end coord is 0
    else:
        last_col = tensor_coords[..., -1:]
        is_zero_mask = torch.isclose(last_col, torch.tensor(0.0, dtype=tensor_coords.dtype, device=tensor_coords.device), atol=1e-7)
        effective_final_coords = torch.where(is_zero_mask, 1.0, last_col)

        final_tensor = tensor_coords[..., :-1] / effective_final_coords
    
    return final_tensor.detach().cpu().numpy() if is_numpy else final_tensor 


def create_projection_matrix(simplified_intrinsics: np.ndarray, extrinsics: np.ndarray):
    return np.matmul(_expand_intrinsics_matrix(simplified_intrinsics), extrinsics) # https://visionbook.mit.edu/imaging_geometry.html#full-camera-model

# MARK: Sample Camera Config
def sample_camera_config(rng: np.random.Generator, base_intrinsics: np.ndarray, base_extrinsics: np.ndarray, sampling_settings=CameraConfigSamplingSettings()):
    """
    Perturbs a base camera's intrinsics and extrinsics
    - intrinsics_jitter - max abs pixels of noise added to each of fx, fy, cx, cy
    - translation_jitter - max abs meters of noise added to T
    - rotation_jitter_deg - max abs degrees of noise applied as a small random rotation, composed on top of the base rotation
    """
    intrinsics_jitter = sampling_settings.intrinsics_jitter
    translation_jitter = sampling_settings.translation_jitter
    rotation_jitter_deg = sampling_settings.rotation_jitter_deg
    # Intrinsics
    intrinsics = base_intrinsics.copy()
    intrinsics[0, 0] += rng.uniform(-intrinsics_jitter, intrinsics_jitter) # fx
    intrinsics[1, 1] += rng.uniform(-intrinsics_jitter, intrinsics_jitter) # fy
    intrinsics[0, 2] += rng.uniform(-intrinsics_jitter, intrinsics_jitter) # cx
    intrinsics[1, 2] += rng.uniform(-intrinsics_jitter, intrinsics_jitter) # cy

    # Extrinsics
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

    return np.stack(projected_trajectory, axis=0).astype(np.float64) # Always saved at full precision; downcast happens only when loaded (see dataset.py)


if __name__ == "__main__":
    paths = Paths()

    intrinsics, extrinsics = get_base_camera_config()

    print(extrinsics if extrinsics is not None else "Failure to obtain extrinsics")

    if extrinsics is not None:
        trajectory, _, _ = generate_sample_trajectory()
        projection_matrix = create_projection_matrix(simplified_intrinsics=intrinsics, extrinsics=extrinsics)
        projected_trajectory = project_trajectory_to_2d(trajectory, projection_matrix)

        overlay_trajectory_on_image(image_path="./sample_img.png", projected_trajectory=projected_trajectory, output_path="./sample_img_overlay.png")

        plot_2d_trajectory(projected_trajectory)


