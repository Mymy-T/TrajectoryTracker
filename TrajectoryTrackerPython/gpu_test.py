import os
import torch

# 1. Count physical gpus recognized by macOS
gpu_count = os.popen("system_profiler SPDisplaysDataType | grep -c 'Chipset Model'").read().strip()

# 2. Check if PyTorch can utilize it via Metal Performance Shaders (MPS)
mps_available = torch.backends.mps.is_available()

print(f"Total gpus found on system: {gpu_count}")
print(f"Pytorch can use gpu: {mps_available}")
