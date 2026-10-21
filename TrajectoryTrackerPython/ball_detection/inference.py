import argparse
from ball_detection.frame_extractor import extract_clip_frames
from ball_detection.model import TrackNet
from ball_detection.paths import ProjectPaths
from ball_detection import settings
import cv2
import os
import numpy as np
import textwrap
import torch
from tqdm import trange

# TODO: Create an mp4 video of the heatmaps too (before threshold is applied)

device = settings.device


def main():
    parser = build_parser()
    args = parser.parse_args()

    paths = ProjectPaths()
    model_name = args.model

    # Validate before creating anything, so a typo doesn't leave an empty run folder behind
    validate_model(parser, paths, model_name)
    if args.existing_frames:
        validate_existing_frames(parser, paths, args.name)
    else:
        validate_source_video(parser, paths, args.name)

    inference_video_folder = os.path.join(paths.inference_results_folder, args.name)
    frames_folder = os.path.join(inference_video_folder, "frames")

    if not os.path.exists(inference_video_folder):
        os.mkdir(inference_video_folder)

    if not os.path.exists(frames_folder):
        os.mkdir(frames_folder)

    if not args.existing_frames:
        original_video_filename = f"{args.name}.mp4"
        extract_clip_frames(video_path=os.path.join(paths.inference_videos_folder, original_video_filename), output_dir=frames_folder)
        print("Frames extracted")

    tracked_ball_points = obtain_tracked_ball_data(inference_video_folder=inference_video_folder, model_name=model_name)
    create_heatmap_video(inference_video_folder=inference_video_folder)
    create_annotated_video(inference_video_folder=inference_video_folder, tracked_ball_points=tracked_ball_points)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="inference.py",
        description="Run a trained TrackNet model over a clip's frames and write out a ball-tracking annotated video and a heatmap video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # extract frames from videos/inference/<name>.mp4, then run
              python3 inference.py single_match_model "Federer_00-00-00_to_00-00-02"

              # reuse frames already extracted into inference_results/<name>/frames
              python3 inference.py single_match_model "Federer_00-00-00_to_00-00-02" --existing-frames

            All output is written to inference_results/<name>/ (frames, heatmaps,
            heatmap_video.mp4, annotated_video.mp4).

            NOTE: clip names usually contain "&" and other characters the shell treats
            specially, so keep NAME in quotation marks.
            """),
    )
    parser.add_argument(
        "model",
        metavar="MODEL",
        help="Name of the trained model, as saved in the model_saves/final_models folder. Do not include the .pt extension, it's added automatically.",
    )
    parser.add_argument(
        "name",
        metavar="NAME",
        help="Name of this inference run. Its results go in inference_results/NAME/. Unless --existing-frames is given, this is ALSO the source clip's filename (without extension), read from videos/inference/NAME.mp4.",
    )
    parser.add_argument(
        "--existing-frames",
        action="store_true",
        help="Reuse the frames already in inference_results/NAME/frames instead of extracting them from a source video. Use this to re-run a model over a clip you've already processed.",
    )
    return parser


def validate_model(parser, paths, model_name):
    """Resolve model_name"""
    stem, extension = os.path.splitext(model_name)
    if extension == "":
        extension = ".pt"
    model_path = os.path.join(paths.final_model_save_folder, stem, stem + extension)

    if not os.path.isfile(model_path):
        available = sorted(
            entry for entry in os.listdir(paths.final_model_save_folder)
            if not entry.startswith(".") and os.path.isdir(os.path.join(paths.final_model_save_folder, entry))
        )
        parser.error(
            f"No model named '{model_name}' (looked for {model_path})\n"
            f"Available models: {', '.join(available) if available else '(none trained yet)'}"
        )


def validate_source_video(parser, paths, name):
    """Check if clip to extract frames from exists"""
    video_path = os.path.join(paths.inference_videos_folder, f"{name}.mp4")

    if not os.path.isfile(video_path):
        available = sorted(
            os.path.splitext(entry)[0] for entry in os.listdir(paths.inference_videos_folder)
            if entry.endswith(".mp4")
        )
        parser.error(
            f"No video at {video_path}\n"
            f"Available inference videos: {', '.join(available) if available else '(none downloaded yet)'}\n"
            f"Download one using video_downloader.py"
        )


def validate_existing_frames(parser, paths, name):
    """Check that the frames to reuse are actually there"""
    frames_folder = os.path.join(paths.inference_results_folder, name, "frames")

    if not os.path.isdir(frames_folder) or not [f for f in os.listdir(frames_folder) if not f.startswith(".")]:
        available = sorted(
            entry for entry in os.listdir(paths.inference_results_folder)
            if not entry.startswith(".") and os.path.isdir(os.path.join(paths.inference_results_folder, entry))
        )
        parser.error(
            f"--existing-frames was given but there are no frames in {frames_folder}\n"
            f"Existing inference runs: {', '.join(available) if available else '(none yet)'}\n"
            f"Drop --existing-frames to extract them from videos/inference/{name}.mp4 instead"
        )


def load_model(model_name):
    paths = ProjectPaths()
    model_name, extension = os.path.splitext(model_name)
    folder_name = model_name
    if extension == "":
        extension = ".pt"
    model_name_with_extension = model_name + extension
    model_path = os.path.join(paths.final_model_save_folder, folder_name,  model_name_with_extension)
    model = TrackNet()
    model_state = torch.load(model_path, weights_only=True, map_location=torch.device(device))
    model.load_state_dict(model_state)
    model.to(device)
    return model


def create_frame_sequence(i, seq_len=3):
    if i < seq_len - 1:
        return None
    frame_ids = list(range(i, i - seq_len, -1)) # [i, i-1, ..., i-(seq_len-1)]; newest first, matching the pretrained model
    return frame_ids


def get_input(img_paths):
        imgs = []
        for img_path in img_paths:
            img = cv2.imread(img_path) # Keep in BGR format, the pretrained model was trained this way
            img = cv2.resize(img, (640, 360)) # Originally 1920, 1080
            imgs.append(img) # Each img of shape (h, w, c)
        
        imgs = np.concatenate(imgs, axis=2) # Concatenate along channel dimension; channels 0-2 are the current frame, higher channels are earlier frames (pretrained model convention)
        imgs = np.moveaxis(imgs, 2, 0) # Shape now (c, h, w)
        imgs = imgs.astype(np.float32) / 255.0 # Normalize pixel values to [0, 1] as done in training
        return imgs


def obtain_tracked_ball_data(inference_video_folder, model_name):
    model = load_model(model_name) # Load the trained model
    model.eval()
    frames_path = os.path.join(inference_video_folder, "frames")

    frames = [frame for frame in os.listdir(frames_path) if not frame.startswith(".")] # All frames have a name of the form frame_########.jpg; ignore hidden files such as .DS_Store
    ball_points = {}

    heatmap_folder = os.path.join(inference_video_folder, "heatmaps")
    if not os.path.exists(heatmap_folder):
        os.mkdir(heatmap_folder)

    with torch.inference_mode():
        for i in trange(len(frames), desc="Obtaining ball positions..."):
            frame_ids = create_frame_sequence(i)

            if frame_ids is None:
                heatmap = np.zeros((360, 640))# Write an empty intermediate heatmap
                cv2.imwrite(os.path.join(heatmap_folder, f"frame_{i:08d}_heatmap.png"), heatmap) # Save intermediate heatmaps
                continue

            frame_ids = list(map(lambda x: os.path.join(frames_path, f"frame_{x:08d}.jpg"), frame_ids))
            model_input = np.expand_dims(get_input(frame_ids), axis=0) # Add batch dimension, batch size just equals 1
            model_input_tensor = torch.from_numpy(model_input).float().to(device)
            model_output_tensor = model(model_input_tensor, testing=True) # Shape is now (N, c, h, w)
            model_output = model.process_output(model_output_tensor) # Shape is now (1, h, w) since batch size (N) = 1
            model_output = np.squeeze(model_output, axis=0) # Shape is now (h, w); 2d array required for cv2.HoughCircles
            cv2.imwrite(os.path.join(heatmap_folder, f"frame_{i:08d}_heatmap.png"), model_output) # Save intermediate heatmaps
            model_output = model.filter_by_threshold(model_output)
            ball_point = model.obtain_ball_center_coords(filtered_output=model_output)

            if ball_point is not None:
                scale_factor = 1080/360 # Multiply center coords by a scale factor to get from 360 x 640 to 1080 x 1920
                ball_point = list(map(lambda x: int(x*scale_factor), ball_point)) # ints used so it can be used when OpenCV draws circles
                ball_points[i] = ball_point
    return ball_points


def create_heatmap_video(inference_video_folder):
    # Video saved to the same folder as the frames (not heatmaps)
    output_path = os.path.join(inference_video_folder, "heatmap_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Standard MP4 codec
    output_fps = settings.inference_fps
    frame_size = (1920, 1080)
    video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, frame_size)

    heatmap_folder_path = os.path.join(inference_video_folder, "heatmaps")
    heatmaps = [heatmap for heatmap in os.listdir(heatmap_folder_path) if not heatmap.startswith(".")] # All heatmaps have a name of the form frame_########.png; ignore hidden files such as .DS_Store
    for i in trange(len(heatmaps), desc="Generating output heatmap video..."):
        heatmap_path = os.path.join(heatmap_folder_path, f"frame_{i:08d}_heatmap.png")
        heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
        heatmap = cv2.resize(heatmap, (1920, 1080))
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        video_writer.write(heatmap)
    video_writer.release()
    print("Heatmap video saved successfully!")


def create_annotated_video(inference_video_folder, tracked_ball_points):
    # Video saved to the same folder as the frames
    paths = ProjectPaths()
    output_path = os.path.join(inference_video_folder, "annotated_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Standard MP4 codec
    output_fps = settings.inference_fps
    frame_size = (1920, 1080)
    video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, frame_size)

    frame_folder_path = os.path.join(inference_video_folder, "frames")
    frames = [frame for frame in os.listdir(frame_folder_path) if not frame.startswith(".")] # All frames have a name of the form frame_########.jpg; ignore hidden files such as .DS_Store
    for i in trange(len(frames), desc="Generating annotated video..."):
        frame_path = os.path.join(frame_folder_path, f"frame_{i:08d}.jpg")
        frame = cv2.imread(frame_path)
        frame_annotations = tracked_ball_points.get(i, None)
        if frame_annotations is not None:
            # Draw tracking circle
            output_frame = frame.copy()
            x, y, r = frame_annotations
            cv2.circle(output_frame, center=(x, y), radius=r, color=(0, 0, 255), thickness=-1) # color in BGR format, thickness of -1 creates a filled in circle
            video_writer.write(output_frame)
        else:
            video_writer.write(frame)

    video_writer.release()
    print("Annotated video saved successfully!")


if __name__ == "__main__":
    main()