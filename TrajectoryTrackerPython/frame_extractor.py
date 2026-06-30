import cv2
import os
from paths import ProjectPaths
import sys


def process_all_videos(video_folder_path, output_folder_path):
    videos = [video for video in os.listdir(video_folder_path) if not video.startswith('.')]
    for i in range(len(videos)):
        try:
            clip_folder = os.path.join(output_folder_path, f"clip_{i:04d}")
            os.mkdir(clip_folder)
        except:
            print(f"Make sure {output_folder_path} is empty beforehand.")
            raise
        extract_all_frames(os.path.join(video_folder_path, videos[i]), clip_folder)

def extract_all_frames(video_path, output_dir):
    vidcap = cv2.VideoCapture(video_path)
    success, img = vidcap.read()
    count = 0
    
    while success:
        cv2.imwrite(f"{output_dir}/frame_{count:08d}.jpg", img)
        success, img = vidcap.read()
        count += 1
        
    vidcap.release()

# Old main
# if __name__ == "__main__":
#     # arg1 should be the video filenmae, arg2 should be output foldername
#     paths = ProjectPaths()
#     video_folder_path = paths.clips_folder
#     output_parent_path = paths.raw_frame_folder

#     if len(sys.argv) != 3:
#         print("Please enter 2 args: video filename and output foldername")
#         sys.exit(1)

    
#     video_path = os.path.join(video_folder_path, sys.argv[1])
#     print(video_path)

#     if not os.path.isdir(sys.argv[2]):
#         os.mkdir(os.path.join(output_parent_path, sys.argv[2]))
#     output_path = os.path.join(output_parent_path, sys.argv[2])

#     extract_all_frames(video_path, output_path)
#     print("Frame extraction complete!")

if __name__ == "__main__":
    paths = ProjectPaths()
    process_all_videos(paths.videos_folder, paths.raw_frame_folder)
    print("Success!")

