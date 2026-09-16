import os
from paths import ProjectPaths
import re
import sys
import yt_dlp

# NOTE: The code may fail with ffmpeg exit code 1, possibly due to internet connection errors. Just keep retrying the same command.

TRAIN_FPS = "60" # Training clips are pinned to this fps to keep the training set frame rate consistent

if len(sys.argv) != 6:
    print("Requires 5 args: video_url start_time end_time fps download_mode")
    print("Eg: python3 video_downloader.py https://youtube.com... 00:02:15 00:05:30 60 train")
    sys.exit(1)

video_url = sys.argv[1]
start_time = sys.argv[2]
end_time = sys.argv[3]
fps = sys.argv[4]
download_mode = sys.argv[5]

if fps not in ("30", "60"):
    print(f"fps must be either 30 or 60, got: {fps}")
    sys.exit(1)

if download_mode not in ("train", "inference"):
    print(f"download_mode must be either train or inference, got: {download_mode}")
    sys.exit(1)

project_paths = ProjectPaths()

if download_mode == "train":
    if fps != TRAIN_FPS:
        print(f"train mode: forcing fps={TRAIN_FPS} (TRAIN_FPS), ignoring requested {fps}")
    fps = TRAIN_FPS
    destination_folder = project_paths.train_videos_folder
else:
    destination_folder = project_paths.inference_videos_folder

start_time_filepart = start_time.replace(":", "-")
end_time_filepart = end_time.replace(":", "-")

print("Extracting video metadata...")
meta_opts = {"quiet": True}
with yt_dlp.YoutubeDL(meta_opts) as ydl:
    info_dict = ydl.extract_info(video_url, download=False)
    title = info_dict.get("title", "video")
    title = re.sub(r'[<>:"/\\|?*]', "_", title)


actual_final_path = os.path.join(
    destination_folder,
    f"{title}_{start_time_filepart}_to_{end_time_filepart}.mp4"
)

print(f"Fetching and slicing clip from {start_time} to {end_time}...")

ydl_opts = {
    "format": f"bestvideo[height=1080][fps={fps}][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]", # NOTE: This will throw an error if any of the conditions aren't met.
    "merge_output_format": "mp4",
    "outtmpl": actual_final_path,
    "quiet": True,
    "force_keyframes_at_cuts": True,
    
    # Download full chunks then slice with ffmpeg
    "postprocessor_args": {
        "ffmpeg": [
            "-ss", start_time,
            "-to", end_time,
            "-c:v", "libx264",    # Re-encode video to clean, universally readable H.264
            "-c:a", "aac"         # Clean audio codec
        ]
    }
}


try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
except yt_dlp.utils.DownloadError:
    print("\nAvailable video formats:\n")

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(video_url, download=False)

    for f in info["formats"]:
        if f.get("vcodec") == "none":
            continue  # Skip audio-only streams

        print(
            f"id={f['format_id']:>4}  "
            f"{f.get('height')}p  "
            f"{f.get('fps')}fps  "
            f"{f.get('vcodec')}  "
            f"{f.get('ext')}"
        )
    
    print("Failure to download clip.")
    sys.exit()

print(f"Success! Saved clip in {actual_final_path}")
