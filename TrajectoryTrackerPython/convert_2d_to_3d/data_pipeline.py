import argparse
from convert_2d_to_3d.dataset_creator import create_dataset
from convert_2d_to_3d.data_splitter import split_data
from convert_2d_to_3d.paths import Paths
from convert_2d_to_3d.trajectory_generator import generate_raw_trajectories
import os

def main():
    parser = argparse.ArgumentParser(description="Generate all of the necessary synthetic data prior to training.")
    parser.add_argument("--physics", action="store_true", help="Regenrate the physics trajectories in addition to everything else.")
    args = parser.parse_args()

    if args.physics:
        generate_raw_data()

    generate_derived_data()

    print("DATA GENERATION SUCCESS!")
    


def generate_raw_data():

    raw_data_path = Paths().raw_data_folder

    if len(os.listdir(raw_data_path)) != 0:
        raise FileExistsError(f"There is already existing raw data at {raw_data_path}. Either delete it, or save a copy somewhere before running this command.")
    
    generate_raw_trajectories()


def generate_derived_data():

    raw_data_path = Paths().raw_data_folder

    if len(os.listdir(raw_data_path)) == 0:
        raise FileExistsError(f"There is no existing raw data at {raw_data_path}. Either run the command with --physics, or insert preexisting saved raw data.")
    
    create_dataset()
    split_data()


if __name__ == "__main__":
    main()