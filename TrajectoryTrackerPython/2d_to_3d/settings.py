from dataclasses import dataclass, field
from data_models import TrajectoryGenerationSettings, CameraConfigSamplingSettings, TrainingConfig
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
target_fps = 30.0 # Simulated trajectories interpolated to this fps; 1/target_fps must be >= simulation_timestep

class CameraPerspectiveMode(Enum):
    BROADCAST = 1
    EYE_LEVEL = 2

class WorldCoordSet(Enum):
    SERVICE_LINE = 1
    INNER_BASELINE = 2 

class NetPostType(Enum):
    SINGLES = 1
    DOUBLES = 2

class DataSplitMode(Enum):
    camera_config_border = 0 # Split only at different camera configs
    no_border = 1 # Split anywhere


@dataclass(frozen=True)
class SyntheticDataGeneration:
    num_camera_configs: int = 3
    trajectories_per_config: int = 5

    trajectory_generation_settings: TrajectoryGenerationSettings = field(default_factory=lambda: TrajectoryGenerationSettings(racket_hits=True))
    camera_config_sampling_settings: CameraConfigSamplingSettings = field(default_factory=lambda: CameraConfigSamplingSettings(intrinsics_jitter=50.0, translation_jitter=0.5, rotation_jitter_deg=0.5))

    camera_perspective_mode: CameraPerspectiveMode = CameraPerspectiveMode.BROADCAST
    camera_perspective_name: str = field(init=False)
    intrinsics_known: bool = field(init=False)

    delayed_hit_secs_after_bounce: float = 0.4 # Amount of time (in s) after a bounce before the next hit should occur if the hit is delayed due to being below the min hit height
    min_hit_height_frac: float = 0.2 # This is the minimum height as a fraction of the max bounce hit below which a hit is delayed until after a bounce


    expected_image_size = (1920, 1080) # (width, height); theoretically enforced by ball_detection's video_downloader.py
    corner_keypoint_set: WorldCoordSet = WorldCoordSet.INNER_BASELINE
    net_post_type: NetPostType = NetPostType.SINGLES # Change based on sample img


    # Data Splitting
    split_mode: DataSplitMode = DataSplitMode.camera_config_border
    train_prop: float = 0.7
    valid_prop: float = 0.15
    split_seed: int = 0

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
    bounce_token_nms_radius: int = 3 # For non-maximum supression
    bounce_token_min_prob: float = 0.9


@dataclass
class Training:
    train_config: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        lr=1e-5,
        batch_size=8,
        max_epochs=10,
        name="", # Temp, passed in when script called
        resuming=False, # Same as name
        eot_imbalance_weight=48.1026,
        bounce_token_imbalance_weight = 38.8958,
        max_train_seconds=None, # Same as name
        max_checkpoints=None # Same as name
    ))
    


