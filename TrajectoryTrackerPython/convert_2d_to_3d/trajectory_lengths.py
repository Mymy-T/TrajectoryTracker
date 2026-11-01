import argparse
from convert_2d_to_3d.paths import Paths
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="A script to view histograms and percentiles/mean of trajectory length of a dataset")
    parser.add_argument("--dataset", required=True, choices=["train", "valid", "test"], help="The dataset to analyze. Either train, valid, or test.")
    args = parser.parse_args()

    data_path = os.path.join(Paths().dataset_csv_folder, f"{args.dataset}_data.csv")
    df = pd.read_csv(data_path)

    trajectory_lengths = df["num_points"].to_list()

    plot_lengths(trajectory_lengths, args.dataset)


def plot_lengths(trajectory_lengths, dataset_type):

    n_bins = floor(len(trajectory_lengths) / 30)

    plt.figure(figsize=(10, 6))
    plt.hist(trajectory_lengths, bins=n_bins)
    plt.xlabel("Trajectory Length")
    plt.ylabel("Frequency")
    plt.title(f"Histogram ({dataset_type.capitalize()} Data)")

    p0, p25, p50, p75, p100 = np.percentile(np.array(trajectory_lengths), [0, 25, 50, 75, 100])
    mean = np.mean(np.array(trajectory_lengths))
    percentile_text = f"Min: {p0}   |   25th Percentile: {p25}   |   Median (50th): {p50}   |   75th Percentile: {p75}   |   Max: {p100}"
    num_text = f"Num Examples: {len(trajectory_lengths)}   |   Num Bins: {n_bins}"
    mean_text = f"Mean: {mean:.2f}"
    plt.figtext(0.5, 0.017, percentile_text, ha="center")
    plt.figtext(0.75, 0.9, mean_text)
    plt.figtext(0.05, 0.9, num_text)


    plt.show()

if __name__ == "__main__":
    main()