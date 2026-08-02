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

print(f"Fetching and slicing clip from {start_time} to {end_time}...")

ydl_opts = {
    "format": "bestvideo[height=1080][fps=60][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]", # NOTE: This will throw an error if any of the conditions aren't met.
    "merge_output_format": "mp4",
    "outtmpl": actual_final_path,
    "quiet": True,
    "force_keyframes_at_cuts": True,
    
    # Download full chunksthen slice with ffmpeg
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
            f"id={f["format_id"]:>4}  "
            f"{f.get("height")}p  "
            f"{f.get("fps")}fps  "
            f"{f.get("vcodec")}  "
            f"{f.get("ext")}"
        )
    
    print("Failure to download clip.")
    sys.exit()

print(f"Success! Saved clip in {actual_final_path}")
