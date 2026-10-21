import argparse
from convert_2d_to_3d.paths import Paths
from convert_2d_to_3d.settings import SyntheticDataGeneration, CameraPerspectiveMode
import os

def main():

    parser = argparse.ArgumentParser(description="Deletes the derived intrinsics/extrinsics (only extrinsics for eye level camera perspective), the labelled keypoints, or both. The sample images are not removed.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--params", action='store_true', help="Remove only the derived camera parameters of the currently selected camera perspective setting.")
    group.add_argument("--keypoints", action='store_true', help="Remove only the labelled keypoints of the currently selected camera perspective setting.")
    group.add_argument("--all", action='store_true', help="Remove both the derived camera parameters and the labelled keypoints of the currently selected camera perspective setting.")
    args = parser.parse_args()

    paths = Paths()
    keypoints_path = paths.saved_keypoints_path
    intrinsics_path = paths.base_intrinsics_path
    extrinsics_path = paths.base_extrinsics_path

    data_gen_settings = SyntheticDataGeneration()
    cam_mode = data_gen_settings.camera_perspective_mode

    if args.params == True:
        reset_params = True
        reset_keypoints = False
    elif args.keypoints == True:
        reset_params = False
        reset_keypoints = True
    else:
        reset_params = True
        reset_keypoints = True


    if reset_params:

        if cam_mode == CameraPerspectiveMode.BROADCAST: # Reset both intrinsics & extrinsics
            os.remove(intrinsics_path)
        
        os.remove(extrinsics_path)

    if reset_keypoints:
        os.remove(keypoints_path)


if __name__ == "__main__":
    main()