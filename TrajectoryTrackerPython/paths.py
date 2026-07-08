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
        self.data_folder = os.path.join(os.getcwd(), "annotated_data") # For annotated data; contains both raw images and labels
        self.videos_folder = os.path.join(os.getcwd(), "videos") # For unedited and long videos (may be downloaded from internet)
        self.processed_data_folder = os.path.join(os.getcwd(), "processed_data") # For processed data ready to be split
