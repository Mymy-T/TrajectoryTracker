import argparse
from dataset import TrajectoryReconstructionDataset, DatasetType, collate_func
from model import FullBounceNetwork
import os
from paths import Paths
import settings
import torch
from torch.utils.data import DataLoader
from trainer import compute_classification_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate the classification metrics of a bounce model.")
    parser.add_argument("--name", required=True, help="The name of the model. This should be a simple string, without extensions. Do NOT add _bm at the end, it will be added automatically.")
    parser.add_argument("--buffer", required=True, type=int, help="The buffer where a model prediction is still considered to be the same as a label and counted right.")
    
    args = parser.parse_args()

    name = args.name
    model_path = os.path.join(Paths().model_saves_folder, f"{name}_bm", f"{name}_bm.pt")
    model = FullBounceNetwork()
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(settings.device)

    test_data = TrajectoryReconstructionDataset(DatasetType.test, noise_2d=True)
    running_on_cpu = (settings.device.type == "cpu")
    test_dataloader = DataLoader(test_data, batch_size=settings.Training().train_config.batch_size, pin_memory=(not running_on_cpu), collate_fn=collate_func)

    accuracy, precision, recall, f1 = compute_classification_metrics(model, test_dataloader, args.buffer)

    print(f"Accuracy: {100 * accuracy:.4f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")



if __name__ == "__main__":
    main()