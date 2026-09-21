import os
from settings import SyntheticDataGeneration

class Paths:
    def __init__(self):

        data_gen_settings = SyntheticDataGeneration()

        base_folder = os.path.dirname(os.path.abspath(__file__)) # paths.py should be on same level as other files

        self.raw_data_folder = os.path.join(base_folder, "raw_data") # Simulated trajectories at the simulation timestep; camera independent
        self.processed_data_folder = os.path.join(base_folder, "processed_data") # Resampled to the target frame rate and projected through a camera
        self.dataset_csv_folder = os.path.join(base_folder, "datasets")

        camera_perspective_name = data_gen_settings.camera_perspective_name

        intrinsics_folder = os.path.join(base_folder, "intrinsics")
        self.base_intrinsics_path = os.path.join(intrinsics_folder, f"{camera_perspective_name}_intrinsics.json") # intrinsics being saved as json is artifact from exporting intrinsics from iPhone camera

        extrinsics_folder = os.path.join(base_folder, "extrinsics")
        self.base_extrinsics_path = os.path.join(extrinsics_folder, f"{camera_perspective_name}_extrinsics.npy")


        camera_calibration_folder = os.path.join(base_folder, "cam_calibration")
        # self.keypoints_folder = os.path.join(base_folder, "cam_calibration")
        self.sample_image_path = os.path.join(camera_calibration_folder, camera_perspective_name, "sample_img.png")
        self.saved_keypoints_path = os.path.join(camera_calibration_folder, camera_perspective_name, "keypoints.npy")

        self.model_saves_folder = os.path.join(base_folder, "model_saves")
        self.logs_folder = os.path.join(base_folder, "logs")

        # TODO: Move the sample_img.png into the keypoints folder; each img/keypoint set should be its own subfolder within keypoints folder


        if not os.path.exists(self.raw_data_folder):
            os.mkdir(self.raw_data_folder)

        if not os.path.exists(self.processed_data_folder):
            os.mkdir(self.processed_data_folder)

        if not os.path.exists(self.model_saves_folder):
            os.mkdir(self.model_saves_folder)

        if not os.path.exists(self.logs_folder):
            os.mkdir(self.logs_folder)
        
        if not os.path.exists(intrinsics_folder):
            os.mkdir(intrinsics_folder)
        
        if not os.path.exists(extrinsics_folder):
            os.mkdir(extrinsics_folder)

        if not os.path.exists(camera_calibration_folder):
            os.mkdir(camera_calibration_folder)

        if not os.path.exists(os.path.join(camera_calibration_folder, camera_perspective_name)):
            os.mkdir(os.path.join(camera_calibration_folder, camera_perspective_name))
        

if __name__ == "__main__":
    paths = Paths()