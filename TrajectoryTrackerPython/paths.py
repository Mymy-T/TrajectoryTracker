import os

class ProjectPaths:
    def __init__(self):
        self.raw_frame_folder = os.path.join(os.getcwd(), "raw_frames") # For frames exported from clips
        # self.clips_folder = os.path.join(os.getcwd(), "clips") # For short videos ready to have their frames exported
        self.data_folder = os.path.join(os.getcwd(), "data") # For annotated data; contains both raw images and labels
        self.videos_folder = os.path.join(os.getcwd(), "videos") # For unedited and long videos (may be downloaded from internet)
        self.train_data_folder = os.path.join(os.getcwd(), "train_data") # For processed data ready to be used for training
