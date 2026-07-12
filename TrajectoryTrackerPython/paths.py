import os
import sys

class ProjectPaths:
    def __init__(self):
        # Ensure the user is actually running this from the script's folder; paths.py should be at the same level and have the same parent folder as all other runnable .py files
        if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
            print("Error: Script must be executed from inside its own folder.")
            print(f"Current directory: {os.getcwd()}")
            print(f"Should be executed within: {os.path.dirname(os.path.abspath(__file__))}")
            sys.exit(1)

        # TODO: make paths if not already exist
        self.raw_frame_folder = os.path.join(os.getcwd(), "raw_frames") # For frames exported from clips
        self.annotated_data_folder = os.path.join(os.getcwd(), "annotated_data") # For annotated data; contains both raw images and labels
        self.videos_folder = os.path.join(os.getcwd(), "videos") # For unedited and long videos (may be downloaded from internet)
        self.processed_data_folder = os.path.join(os.getcwd(), "processed_data") # For processed data ready to be split
        self.dataset_folder = os.path.join(os.getcwd(), "datasets") # For .csv files listing the processed data paths to be read when loading for training; contains train, valid, and test csv files

        if not os.path.exists(self.raw_frame_folder):
            os.mkdir(self.raw_frame_folder)
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)
        if not os.path.exists(self.videos_folder):
            os.mkdir(self.videos_folder)
        if not os.path.exists(self.processed_data_folder):
            os.mkdir(self.processed_data_folder)
        if not os.path.exists(self.dataset_folder):
            os.mkdir(self.dataset_folder)

