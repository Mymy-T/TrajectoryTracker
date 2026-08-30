from dataclasses import dataclass, field
from data_models import TrajectoryGenerationSettings, CameraConfigSamplingSettings
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@dataclass(frozen=True)
class SyntheticDataGeneration:
    num_camera_configs: int = 10
    trajectories_per_config: int = 50
    trajectory_generation_settings: TrajectoryGenerationSettings = field(default_factory=lambda: TrajectoryGenerationSettings(
        racket_hits=True
    ))
    camera_config_sampling_settings: CameraConfigSamplingSettings = field(default_factory=lambda: CameraConfigSamplingSettings())

