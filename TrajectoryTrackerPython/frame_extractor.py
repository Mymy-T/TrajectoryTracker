import cv2
import os
from paths import ProjectPaths


def process_all_videos(video_folder_path, output_folder_path):
    videos = [video for video in os.listdir(video_folder_path) if not video.startswith('.')]

    match_groups = {}
    for video in videos:
        match_title = get_match_title(video)
        if match_title not in match_groups:
            match_groups[match_title] = []
        match_groups[match_title].append(video)
    
     # Process each match group using just the unique title
    for match_title, match_videos in match_groups.items():
        # Keep only alphanumeric characters and underscores to ensure a valid folder name
        clean_title = "".join([c if c.isalnum() or c == "_" else "" for c in match_title])

        # Simple, clean folder name using just the title
        match_folder_name = f"{clean_title}"
        match_folder = os.path.join(output_folder_path, match_folder_name)
        
        os.makedirs(match_folder, exist_ok=True)

        # Sort clips chronologically by filename (timestamp order)
        for clip_index, video_file in enumerate(sorted(match_videos)):
            clip_folder = os.path.join(match_folder, f"clip_{clip_index:04d}")
            if os.path.exists(clip_folder):
                # Skip extraction if this clip folder already exists from a previous run
                print(f"Skipping existing clip: {match_folder_name}/clip_{clip_index:04d}")
                continue
                
            print(f"Extracting clip: {video_file} -> {match_folder_name}/clip_{clip_index:04d}")
            os.makedirs(clip_folder, exist_ok=True)
            video_full_path = os.path.join(video_folder_path, video_file)
            extract_clip_frames(video_full_path, clip_folder)
    

def get_match_title(filename):
    """
    Extracts the base video title by stripping off the trailing timestamp structure.
    Eg: "title_00-02-15_to_00-05-30.mp4" -> "title
    """
    # Remove the .mp4 extension
    base_name = os.path.splitext(filename)[0]
    
    if "_to_" in base_name:
        parts = base_name.split("_to_")
        match_part = parts[0] # ignore the latter half; now of the form "title_XX-XX-XX"
        # Remove the starting timestamp attached to the match title (the last item after splitting by '_')
        if "_" in match_part:
            match_part = match_part.rsplit("_", 1)[0] # After split, you get ["title", "XX-XX-XX"] as it's split once at the rightmost "_" 
        return match_part
        
    print(f"Error with video name: {filename} is not of the form 'title_XX-XX-XX_to_XX-XX-XX'")
    raise

def extract_clip_frames(video_path, output_dir):
    vidcap = cv2.VideoCapture(video_path)
    success, img = vidcap.read()
    count = 0
    
    while success:
        cv2.imwrite(f"{output_dir}/frame_{count:08d}.jpg", img)
        success, img = vidcap.read()
        count += 1
        
    vidcap.release()



if __name__ == "__main__":
    paths = ProjectPaths()
    process_all_videos(paths.videos_folder, paths.raw_frame_folder)
    print("Success!")

