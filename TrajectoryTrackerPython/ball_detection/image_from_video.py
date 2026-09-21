import cv2
import os
from paths import ProjectPaths
import sys


def main():

    if len(sys.argv) != 2:
        print("USAGE: python3 image_from_video.py <file_name_of_inference_video>") # NOTE: video extension is included
        sys.exit(1)
    
    video_name = sys.argv[1]

    paths = ProjectPaths()
    video_path = os.path.join(paths.inference_videos_folder, video_name)

    save_first_frame(video_path)


def save_first_frame(video_path):

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise RuntimeError("Could not open video clip.")
    
    success, frame = capture.read()

    if success:
        save_path = os.path.join(os.getcwd(), "sample_img.png")
        cv2.imwrite(save_path, frame)
        print(f"Successfully saved first frame to {save_path}")
    else:
        raise RuntimeError("Could not obtain first frame of the provided video.")
    
    capture.release()


if __name__ == "__main__":
    main()



# Commands:
# python3 image_from_video.py "Alcaraz & Ruud Practice 2024 _ Court Level Groundstrokes (4K 60FPS)_00-01-51_to_00-02-07.mp4"