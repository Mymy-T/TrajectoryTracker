import os

class ProjectPaths:
    def __init__(self):
        self.video_folder_path = os.path.join(os.getcwd(), "videos")
        self.original_data_folder_path = os.path.join(os.getcwd(), "original_data")