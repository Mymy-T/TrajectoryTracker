from dataclasses import dataclass, field
from data_models import TrajectoryGenerationSettings, CameraConfigSamplingSettings
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

dtype = np.float32 # Controls the dtype of all loaded arrays and tensors in the pipeline; must be np.float16, np.float32, or np.float64

_torch_numpy_dtype_dict = {np.float16: torch.float16, np.float32: torch.float32, np.float64: torch.float64}
if dtype not in _torch_numpy_dtype_dict:
    raise ValueError("settings.dtype must be one of np.float16, np.float32, or np.float64")

torch_dtype = _torch_numpy_dtype_dict[dtype]

estimate_height_from_diameter = False

@dataclass(frozen=True)
class SyntheticDataGeneration:
    num_camera_configs: int = 10
    trajectories_per_config: int = 50
    trajectory_generation_settings: TrajectoryGenerationSettings = field(default_factory=lambda: TrajectoryGenerationSettings(
        racket_hits=True
    ))
    camera_config_sampling_settings: CameraConfigSamplingSettings = field(default_factory=lambda: CameraConfigSamplingSettings())

