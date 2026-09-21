from dataclasses import dataclass
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

train_fps = 60 # fps of training clips; the model is trained on sequences sampled at this rate
inference_fps = 30 # fps of inference clips, and of the annotated/heatmap videos written by inference.py
