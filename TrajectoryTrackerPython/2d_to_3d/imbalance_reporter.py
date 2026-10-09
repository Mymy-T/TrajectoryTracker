import argparse
from enum import StrEnum
import numpy as np
import os
import pandas as pd
from paths import Paths

class ModelType(StrEnum):
    BOUNCE = "Bounce Model"
    EOT = "EOT Model"


def get_stats(model_type: ModelType):
    train_data_path = os.path.join(Paths().dataset_csv_folder, "train_data.csv")
    train_df = pd.read_csv(train_data_path)

    if model_type == ModelType.BOUNCE:
        data_paths = list(train_df["trajectory_bounce_tokens_path"])
    elif model_type == ModelType.EOT:
        data_paths = list(train_df["trajectory_eot_path"])

    data_paths = [os.path.join(Paths().processed_data_folder, rel_path) for rel_path in data_paths]

    total_steps = 0
    positive_steps = 0
    for path in data_paths:
        data = np.load(path)
        total_steps += len(data)
        positive_steps += np.sum(data)

    return total_steps, positive_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the counts of positive/negative tokens, and compute imbalance weights.")
    parser.add_argument("--type", required=True, choices=["bounce", "eot"], help="The type of model (e.g. EOT model).")
    args = parser.parse_args()

    match args.type:
        case "bounce":
            total, positive = get_stats(ModelType.BOUNCE)
        case "eot":
            total, positive = get_stats(ModelType.EOT)
    

    negative = total - positive

    print(f"TYPE: {args.type}")
    print(f"Positive: {positive}")
    print(f"Negative: {negative}")
    print(f"Total: {total}")
    print(f"Imbalance weight (negative / positive): {(negative / positive):.4f}")
    