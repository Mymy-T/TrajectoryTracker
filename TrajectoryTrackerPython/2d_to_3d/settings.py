from dataclasses import dataclass, field
from data_models import TrajectoryGenerationSettings, CameraConfigSamplingSettings
from enum import Enum
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

dtype = np.float32 # Controls the dtype of all loaded arrays and tensors in the pipeline; must be np.float16, np.float32, or np.float64

_torch_numpy_dtype_dict = {np.float16: torch.float16, np.float32: torch.float32, np.float64: torch.float64}
if dtype not in _torch_numpy_dtype_dict:
    raise ValueError("settings.dtype must be one of np.float16, np.float32, or np.float64")

torch_dtype = _torch_numpy_dtype_dict[dtype]

estimate_height_from_diameter = False

simulation_timestep = 0.001
target_fps = 60.0 # Simulated trajectories interpolated to this fps; 1/target_fps must be >= simulation_timestep

class CameraPerspectiveMode(Enum):
    BROADCAST = 1
    EYE_LEVEL = 2

class WorldCoordSet(Enum):
    SERVICE_LINE = 1
    INNER_BASELINE = 2 

class NetPostType(Enum):
    SINGLES = 1
    DOUBLES = 2

@dataclass(frozen=True)
class SyntheticDataGeneration:
    num_camera_configs: int = 10
    trajectories_per_config: int = 50

    trajectory_generation_settings: TrajectoryGenerationSettings = field(default_factory=lambda: TrajectoryGenerationSettings(racket_hits=True))
    camera_config_sampling_settings: CameraConfigSamplingSettings = field(default_factory=lambda: CameraConfigSamplingSettings())

    camera_perspective_mode: CameraPerspectiveMode = CameraPerspectiveMode.BROADCAST
    camera_perspective_name: str = field(init=False)
    intrinsics_known: bool = field(init=False)


    expected_image_size = (1920, 1080) # (width, height); theoretically enforced by ball_detection's video_downloader.py
    corner_keypoint_set: WorldCoordSet = WorldCoordSet.INNER_BASELINE
    net_post_type: NetPostType = NetPostType.SINGLES # Change based on sample img

    def __post_init__(self):

        match self.camera_perspective_mode:
            case CameraPerspectiveMode.BROADCAST:
                name = "broadcast_view"
                intrinsics_known = False
            case CameraPerspectiveMode.EYE_LEVEL:
                name = "eye_level_view"
                intrinsics_known = True

        object.__setattr__(self, "camera_perspective_name", name)
        object.__setattr__(self, "intrinsics_known", intrinsics_known)



@dataclass(frozen=True)
class Model:
    bounce_token_nms_radius: int = 3
    bounce_token_min_prob: int = 0.9