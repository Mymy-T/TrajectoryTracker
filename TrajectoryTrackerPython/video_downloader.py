import os
from paths import ProjectPaths
import re
import sys
import yt_dlp

# NOTE: The code may fail with ffmpeg exit code 1, possibly due to internet connection errors. Just keep retrying the same command.

if len(sys.argv) != 4:
    print("Requires 3 args: video_url start_time end_time")
    print("Eg: python3 video_downloader.py https://youtube.com... 00:02:15 00:05:30")
    sys.exit(1)

video_url = sys.argv[1]
start_time = sys.argv[2]
end_time = sys.argv[3]

project_paths = ProjectPaths()

start_time_filepart = start_time.replace(":", "-")
end_time_filepart = end_time.replace(":", "-")

# Fast metadata extraction
print("Extracting video metadata...")
meta_opts = {"quiet": True}
with yt_dlp.YoutubeDL(meta_opts) as ydl:
    info_dict = ydl.extract_info(video_url, download=False)
    title = info_dict.get("title", "video")
    title = re.sub(r'[<>:"/\\|?*]', "_", title)


actual_final_path = os.path.join(
    project_paths.videos_folder, 
    f"{title}_{start_time_filepart}_to_{end_time_filepart}.mp4"
)

# Download only the required chunk directly from YouTube's server
print(f"Fetching and slicing clip from {start_time} to {end_time}...")

ydl_opts = {
    # H.264 (avc1) codec for full OpenCV compatibility
    "format": "bestvideo[height=1080][fps=60][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]", # NOTE: This will throw an error if any of the conditions aren't met.
    "merge_output_format": "mp4",
    "outtmpl": actual_final_path,
    "quiet": True,
    
    #  ffmpeg to post-process the slice cleanly instead of stream-slicing input
    "force_keyframes_at_cuts": True,
    
    # Download the full stream chunks safely, then cleanly slice with ffmpeg post-processor
    "postprocessor_args": {
        "ffmpeg": [
            "-ss", start_time,
            "-to", end_time,
            "-c:v", "libx264",    # Re-encode video to clean, universally readable H.264
            "-c:a", "aac"         # Clean audio codec
        ]
    }
}


with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

print(f"Success! Saved clip in {actual_final_path}")
