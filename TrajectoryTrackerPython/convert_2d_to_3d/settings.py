from convert_2d_to_3d.data_models import TrajectoryGenerationSettings, CameraConfigSamplingSettings, TrainingConfig, TrajectoryMode
from dataclasses import dataclass, field
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


@dataclass
class SyntheticDataGeneration:

    # NUMBER OF TRAJECTORIES
    num_camera_configs: int = 100
    trajectories_per_config: int = 50

    # PHYSICS GENERATION SETTINGS
    physics_rng: np.random.Generator = np.random.default_rng(seed=67)
    trajectory_generation_settings: TrajectoryGenerationSettings = field(default_factory=lambda: TrajectoryGenerationSettings(
        simulation_timestep=0.001,
        mode=TrajectoryMode.BOUNCE_AND_HIT,
        spin_decay=False,
        oob_buffer=4,
        max_bounces=4,
        bounce_trigger_cap=3,
        num_hits_range=(3, 5),
        hit_zone_bounds=(7/16, 1/2),
        min_hit_height_frac=0.2,
        delayed_hit_secs_after_bounce=0.3
    ))

    # PROJECTION/INTERPOLATION SETTINGS
    camera_rng: np.random.Generator = np.random.default_rng(seed=100)
    camera_perspective_mode: CameraPerspectiveMode = CameraPerspectiveMode.BROADCAST
    camera_perspective_name: str = field(init=False)
    intrinsics_known: bool = field(init=False)
    expected_image_size = (1920, 1080) # (width, height); theoretically enforced by ball_detection's video_downloader.py
    corner_keypoint_set: WorldCoordSet = WorldCoordSet.INNER_BASELINE
    net_post_type: NetPostType = NetPostType.SINGLES # Change based on sample img
    camera_config_sampling_settings: CameraConfigSamplingSettings = field(default_factory=lambda: CameraConfigSamplingSettings(
        intrinsics_jitter=50.0,
        translation_jitter=0.5,
        rotation_jitter_deg=0.5
    ))
    target_fps = 30.0 # Simulated trajectories interpolated to this fps; 1/target_fps must be >= simulation_timestep


    # Data Splitting
    split_seed: int = 1 # Separate seed involved in pandas df sampling
    split_mode: DataSplitMode = DataSplitMode.camera_config_border
    train_prop: float = 0.7
    valid_prop: float = 0.15

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
        lr=0.001,
        batch_size=128,
        max_epochs=300,
        name="", # Temp value, passed in when script called
        resuming=False, # See comment for 'name'
        eot_imbalance_weight=47.2332,
        bounce_token_imbalance_weight = 38.8254,
        loss_component_weights=(10, 1, 10), # Used only when training full model; weights are for EOT Loss, Trajecory Loss, and Below-Ground Loss, respectively; append a number for Height Loss if want to enable it
        max_train_seconds=None, # See comment for 'name'
        max_checkpoints=None # See comment for 'name'
    ))
    
