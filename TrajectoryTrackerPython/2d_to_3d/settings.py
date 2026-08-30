from dataclasses import dataclass, field
from data_models import TrajectoryGenerationSettings, CameraConfigSamplingSettings
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

dtype = np.float32 # Controls the dtype of all loaded arrays and tensors in the pipeline; must be np.float16, np.float32, or np.float64

_TORCH_DTYPE_BY_NUMPY_DTYPE = {np.float16: torch.float16, np.float32: torch.float32, np.float64: torch.float64}
if dtype not in _TORCH_DTYPE_BY_NUMPY_DTYPE:
    raise ValueError("settings.dtype must be one of np.float16, np.float32, or np.float64")

torch_dtype = _TORCH_DTYPE_BY_NUMPY_DTYPE[dtype]

@dataclass(frozen=True)
class SyntheticDataGeneration:
    num_camera_configs: int = 10
    trajectories_per_config: int = 50
    trajectory_generation_settings: TrajectoryGenerationSettings = field(default_factory=lambda: TrajectoryGenerationSettings(
        racket_hits=True
    ))
    camera_config_sampling_settings: CameraConfigSamplingSettings = field(default_factory=lambda: CameraConfigSamplingSettings())

