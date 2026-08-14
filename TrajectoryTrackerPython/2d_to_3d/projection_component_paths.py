import os

class Paths:
    def __init__(self):

        base_folder = os.path.dirname(os.path.abspath(__file__)) # paths.py should be on same level as other files

        self.synthetic_data_folder = os.path.join(base_folder, "data")
        self.dataset_csv_folder = os.path.join(base_folder, "datasets")

        self.base_intrinsics_path = os.path.join(base_folder, "base_intrinsics.json")
        self.sample_image_path = os.path.join(base_folder, "sample_img.png")
        self.saved_keypoints_path = os.path.join(base_folder, "keypoints.npy")


        if not os.path.exists(self.synthetic_data_folder):
            os.mkdir(self.synthetic_data_folder)
        

