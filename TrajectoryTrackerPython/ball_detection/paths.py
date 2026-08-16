import os

class ProjectPaths:
    def __init__(self):
        base_folder = os.path.dirname(os.path.abspath(__file__))

        # Data Collection & Dataset Creation
        self.raw_frame_folder = os.path.join(base_folder, "raw_frames") # For frames exported from clips
        self.annotated_data_folder = os.path.join(base_folder, "annotated_data") # For annotated data; contains both raw images and labels
        self.videos_folder = os.path.join(base_folder, "videos") # For unedited and long videos (may be downloaded from internet)
        self.processed_data_folder = os.path.join(base_folder, "processed_data") # For processed data ready to be split
        self.dataset_folder = os.path.join(base_folder, "datasets") # For .csv files listing the processed data paths to be read when loading for training; contains train, valid, and test csv files

        # Training Logs
        self.final_model_training_logs = os.path.join(base_folder, "logs", "final_models")
        self.hyperparam_tuning_training_logs = os.path.join(base_folder, "logs", "hyperparam_tuning")

        # Model Paths
        self.final_model_save_folder = os.path.join(base_folder, "model_saves", "final_models")
        self.hyperparam_tuning_model_save_folder = os.path.join(base_folder, "model_saves", "hparam_tune_models")
        self.pretrained_model_path = os.path.join(self.final_model_save_folder, "pretrained_model.pt")

        # Testing
        self.videos_for_inference_folder = os.path.join(base_folder, "inference_videos")

        # Misc
        self.match_title_mapping_path = os.path.join(base_folder, "match_title_to_number_mapping.txt")

        if not os.path.exists(self.raw_frame_folder):
            os.mkdir(self.raw_frame_folder)
        if not os.path.exists(self.annotated_data_folder):
            os.mkdir(self.annotated_data_folder)
        if not os.path.exists(self.videos_folder):
            os.mkdir(self.videos_folder)
        if not os.path.exists(self.processed_data_folder):
            os.mkdir(self.processed_data_folder)
        if not os.path.exists(self.dataset_folder):
            os.mkdir(self.dataset_folder)
        if not os.path.exists(self.videos_for_inference_folder):
            os.mkdir(self.videos_for_inference_folder)

        os.makedirs(self.final_model_training_logs, exist_ok=True)
        os.makedirs(self.hyperparam_tuning_training_logs, exist_ok=True)
        os.makedirs(self.final_model_save_folder, exist_ok=True)
        os.makedirs(self.hyperparam_tuning_model_save_folder, exist_ok=True)
