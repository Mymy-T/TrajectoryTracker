import cv2
from data_models import CourtProperties, CameraConfigSamplingSettings, BallProperties
from enum import Enum
import json
from manual_keypoint_labeler import obtain_keypoints
import math
import numpy as np
import os
from paths import Paths
from physics import generate_sample_trajectory
import settings
import torch
from visualizer import overlay_trajectory_on_image, plot_2d_trajectory


def _generate_3d_baseline_coords(coord_set: settings.WorldCoordSet):
    """
    Returns a ndarray of size 4 x 3; each row is a 3D coord; the rows are ordered as far_left, far_right, near_left, near_right, in that precise order
    """
    court_properties = CourtProperties()
    w = court_properties.court_width
    l = court_properties.court_length
    sw = court_properties.singles_width
    sld = court_properties.service_line_dist_from_net
    if coord_set == settings.WorldCoordSet.SERVICE_LINE:
        far_left = np.array([sld, 0, -sw/2]).astype(np.float32)
        far_right = np.array([sld, 0, sw/2]).astype(np.float32)
        near_left = np.array([-sld, 0, -sw/2]).astype(np.float32)
        near_right = np.array([-sld, 0, sw/2]).astype(np.float32)
    elif coord_set == settings.WorldCoordSet.INNER_BASELINE:
        far_left = np.array([l/2, 0, -sw/2]).astype(np.float32)
        far_right = np.array([l/2, 0, sw/2]).astype(np.float32)
        near_left = np.array([-l/2, 0, -sw/2]).astype(np.float32)
        near_right = np.array([-l/2, 0, sw/2]).astype(np.float32)
    
    return np.stack([far_left, far_right, near_left, near_right], axis=0)

def _generate_full_calibration_3d_coords(coord_set: settings.WorldCoordSet):

    court_properties = CourtProperties()

    corners = _generate_3d_baseline_coords(coord_set) # (4, 3)

    npdfs = court_properties.net_post_distance_from_sideline
    nph = court_properties.net_height_at_posts
    nch = court_properties.net_height_at_center
    sld = court_properties.service_line_dist_from_net
    w = court_properties.court_width
    sw = court_properties.singles_width

    # Net points
    if settings.SyntheticDataGeneration().net_post_type == settings.NetPostType.DOUBLES:
        left_net_post = np.array([0, nph, -(w/2 + npdfs)]).astype(np.float32)
        right_net_post = np.array([0, nph, w/2 + npdfs]).astype(np.float32)
    else:
        left_net_post = np.array([0, nph, -(sw/2 + npdfs)]).astype(np.float32)
        right_net_post = np.array([0, nph, sw/2 + npdfs]).astype(np.float32)
    center_of_net = np.array([0, nch, 0]).astype(np.float32)

    # Extra points at center of "T"
    far_t = np.array([sld, 0, 0]).astype(np.float32)
    near_t = np.array([-sld, 0, 0]).astype(np.float32)

    additional_points = np.stack([left_net_post, center_of_net, right_net_post, far_t, near_t]) # (5, 3)

    return np.concatenate((corners, additional_points), axis=0) # (9, 3)



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

def obtain_base_extrinsics(filepath):
    return np.load(filepath)


def obtain_extrinsics_from_pnp(coord_set: settings.WorldCoordSet, image_points: np.ndarray, intrinsics: np.ndarray, save_path, distortion_coeffs=None, verbose=False):
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

        np.save(save_path, extrinsics)

    return success


def perform_full_camera_calibration(coord_set: settings.WorldCoordSet, image_points: np.ndarray, intrinsics_save_path, extrinsics_save_path, verbose=False):
    """
    - image_points - a 9 x 2 ndarray
    """

    world_points = _generate_full_calibration_3d_coords(coord_set)

    # Initial constraints
    image_shape = settings.SyntheticDataGeneration().expected_image_size
    cx = image_shape[0] / 2.0
    cy = image_shape[1] / 2.0
    f = 1000.0  # Rough guess in pixels; assumed that fx = fy

    intrinsics_guess = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS |
        cv2.CALIB_FIX_ASPECT_RATIO |
        cv2.CALIB_FIX_PRINCIPAL_POINT |
        cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
        cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 |
        cv2.CALIB_FIX_TANGENT_DIST
    )

    rms_error, intrinsics, dist_coefs, r_vecs, t_vecs = cv2.calibrateCamera([world_points], [image_points], image_shape, intrinsics_guess, dist_coeffs, flags=flags) # rvecs and tvecs are lists w/ only one element since only one img was used in calibration; also why world points and image points are passed as lists

    r_vec = r_vecs[0]
    t_vec = t_vecs[0]
    r_matrix, _ = cv2.Rodrigues(r_vec) # Convert rotation vector into 3 x 3 rotation matrix

    if verbose:
        print("Each column of the rotation matrix is a basis vector of the camera coordinate system in terms of world coordinates. The first column is +x (right in camera view), second column is +y (down in camera view), and third column is +z (straight forward in camera view)")
        print(f"Rotation matrix: \n{r_matrix}")
        print(f"Camera position (in world coords): \n{-np.matmul(np.transpose(r_matrix), t_vec)}")

    # t_vec is the position of the world origin from camera POV; therefore, t_vec is in camera frame and t_vec = R(P_w - t) = R(0 - t) = -Rt where R is 3 x 3 and t is 3 x 1 (simplified or heterogeneous form)
    extrinsics = np.identity(4, dtype=np.float64)
    extrinsics[0:3, 0:3] = r_matrix
    extrinsics[0:3, 3] = t_vec.flatten() # Last col of extrinsic matrix is -Rt = t_vec


    # Save intrinsics and extrinsics
    intrinsics_data = {
        "fy": intrinsics[1, 1],
        "cx": intrinsics[0, 2],
        "fx": intrinsics[0, 0],
        "cy": intrinsics[1, 2],
        "referenceWidth": image_shape[0],
        "referenceHeight": image_shape[1]
    }
    with open(intrinsics_save_path, "w") as file:
        json.dump(intrinsics_data, file) # Keep format consistent with iPhone intrinsics exports


    np.save(extrinsics_save_path, extrinsics)

    return rms_error

# MARK: Base Camera Config
def get_base_camera_config():
    paths = Paths()
    data_gen_settings = settings.SyntheticDataGeneration()
    
    # Obtain new keypoints even if intrinsics/extrinsics exist, since the old keypoints can't be found
    keypoint_path = paths.saved_keypoints_path
    if os.path.exists(keypoint_path):
        image_points = np.load(keypoint_path)
        keypoints_found = True
    else:
        obtain_keypoints(verbose=True) # Saves to keypoint_path
        image_points = np.load(keypoint_path)
        keypoints_found = False # Re-calibrate params even if they already exist since they no longer reflect the new keypoints

    if data_gen_settings.intrinsics_known: # Load intrinsics (exported from iPhone), then solve PnP problem for extrinsics

        if not os.path.exists(paths.base_intrinsics_path):
            raise RuntimeError("Known intrinsics must already be formatted as a json in the 'intrinsics' folder.")
        
        base_intrinsics = obtain_base_intrinsics(paths.base_intrinsics_path)

        if not os.path.exists(paths.base_extrinsics_path) or not keypoints_found:
            
            success = obtain_extrinsics_from_pnp(data_gen_settings.corner_keypoint_set, image_points, base_intrinsics, paths.base_extrinsics_path)

            if not success:
                raise RuntimeError("Failed to obtain extrinsics from PnP.")
            
    else: # Full camera calibration
        if not (os.path.exists(paths.base_intrinsics_path) and os.path.exists(paths.base_extrinsics_path)) or not keypoints_found:
            calibration_error = perform_full_camera_calibration(data_gen_settings.corner_keypoint_set, image_points, paths.base_intrinsics_path, paths.base_extrinsics_path)
            print(f"Calibration rms error: {calibration_error:.4f}")
        
        base_intrinsics = obtain_base_intrinsics(paths.base_intrinsics_path)

    base_extrinsics = obtain_base_extrinsics(paths.base_extrinsics_path)

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


def obtain_projected_diameters(trajectory: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray):
    """
    trajectory is ndarray of 3D points
    intrinsics is (3, 3) ndarray
    extrinsics is (4, 4) ndarray
    Returns a (2,) ndarray of the diameters for the initial and final frames
    """

    r = BallProperties().R

    r_matrix = extrinsics[0:3, 0:3] # (3, 3)
    t_vec = extrinsics[0:3, 3:] # (3, 1)
    T = np.squeeze(np.matmul(-r_matrix.T, t_vec), axis=-1) # t_vec = -RT -> T = -R^T t_vec since R is orthogonal; T is (3,)

    def obtain_projected_diameter(S, C):
        """
        Both are (3,); S is the center of ball, C is camera position; everything in world coords
        """

        CS = S - C

        Y = np.array([0, 1, 0]) # Any vector in world coords not parallel to CS works

        c_hat = CS / np.linalg.norm(CS)
        
        D = Y - (np.dot(Y, CS) / np.linalg.norm(CS)) * c_hat
        d_hat = D / np.linalg.norm(D)

        point_1 = S + r * math.sqrt(1 - (r / np.linalg.norm(CS))**2) * d_hat - (r**2 / np.linalg.norm(CS)) * c_hat # Both (3,)
        point_2 = S - r * math.sqrt(1 - (r / np.linalg.norm(CS))**2) * d_hat - (r**2 / np.linalg.norm(CS)) * c_hat

        projection_matrix = create_projection_matrix(intrinsics, extrinsics)
        homogeneous_point_1 = homogenize(point_1)
        homogeneous_point_2 = homogenize(point_2)

        projected_point_1 = dehomogenize(np.matmul(projection_matrix, homogeneous_point_1)) # Both (2,)
        projected_point_2 = dehomogenize(np.matmul(projection_matrix, homogeneous_point_2))

        diameter = np.linalg.norm(projected_point_1 - projected_point_2)
        return diameter
    

    first_point = trajectory[0, :] # Both (3,)
    last_point = trajectory[-1, :]

    d_o = obtain_projected_diameter(S=first_point, C=T)
    d_f = obtain_projected_diameter(S=last_point, C=T)

    return np.array([d_o, d_f]).astype(np.float64) # Always saved at full precision; downcast happens only when loaded (see dataset.py)

if __name__ == "__main__":
    paths = Paths()

    intrinsics, extrinsics = get_base_camera_config()

    print(extrinsics if extrinsics is not None else "Failure to obtain extrinsics")

    if extrinsics is not None:
        r_matrix = extrinsics[0:3, 0:3] # (3, 3)
        t_vec = extrinsics[0:3, 3:] # (3, 1)
        T = np.squeeze(np.matmul(-r_matrix.T, t_vec), axis=-1) # t_vec = -RT -> T = -R^T t_vec since R is orthogonal; T is (3,)
        print(f"Camera position in world coords: {T}")

    if extrinsics is not None:
        trajectory, _, _ = generate_sample_trajectory()
        projection_matrix = create_projection_matrix(simplified_intrinsics=intrinsics, extrinsics=extrinsics)
        projected_trajectory = project_trajectory_to_2d(trajectory, projection_matrix)

        # overlay_trajectory_on_image(image_path="./sample_img.png", projected_trajectory=projected_trajectory, output_path="./sample_img_overlay.png")

        plot_2d_trajectory(projected_trajectory)


