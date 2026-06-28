import cv2
import os
from paths import ProjectPaths
import sys

def extract_all_frames(video_path, output_dir):
    vidcap = cv2.VideoCapture(video_path)
    success, img = vidcap.read()
    count = 0
    
    while success:
        cv2.imwrite(f"{output_dir}/frame_{count:04d}.jpg", img)
        success, img = vidcap.read()
        count += 1
        
    vidcap.release()

if __name__ == "__main__":
    # arg1 should be the video filenmae, arg2 should be output foldername
    paths = ProjectPaths()
    video_folder_path = paths.video_folder_path
    output_parent_path = paths.original_data_folder_path

    if len(sys.argv) != 3:
        print("Please enter 2 args: video filename and output foldername")
        sys.exit(1)

    
    video_path = os.path.join(video_folder_path, sys.argv[1])
    print(video_path)

    if not os.path.isdir(sys.argv[2]):
        os.mkdir(os.path.join(output_parent_path, sys.argv[2]))
    output_path = os.path.join(output_parent_path, sys.argv[2])

    extract_all_frames(video_path, output_path)

