from dataset import TrackNetDataset, DatasetType
from inference import load_model
from model import TrackNet
import os
from paths import ProjectPaths
import sys
from torch.utils.data import DataLoader
import trainer
import settings

device = settings.device


if len(sys.argv) != 2:
    print("USAGE: python3 eval_script.py <final_model_name>")
    sys.exit(1)

paths = ProjectPaths()
model = load_model(sys.argv[1])
test_data = TrackNetDataset(DatasetType.test)
running_on_cpu = (device.type == "cpu")
dataloader = DataLoader(test_data, batch_size=8, pin_memory=(not running_on_cpu))
loss_fn = trainer.FocalLoss(gamma=2.0)
_ = trainer.evaluate(model, loss_fn, dataloader, valid=False, verbose=True)
print("Evaluation complete!")
