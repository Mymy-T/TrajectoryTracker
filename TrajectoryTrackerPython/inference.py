import cv2
from frame_extractor import extract_clip_frames
import os
from model import TrackNet
import numpy as np
from paths import ProjectPaths
import sys
import torch
from tqdm import trange

# TODO: Create an mp4 video of the heatmaps too (before threshold is applied)

# <> -> required input; [] -> optional input
# USAGE: python3 inference.py <model_name> <preexisting_video> [inference_video_folder] [sample_video_name]
# NOTE: inference_video_name and sample_video_name are mutually exclusive; specify inference_video_folder if you set preexisting_video to True; specify sample_video_name if you set preexisting_video to False
# preexisting_video is a bool, and should only take on True or False
# inference_video_folder is the name of the folder in the project path "videos_for_inference" folder (no extension)
# model_name is the name of the trained model, as saved in the project path "final_model_save" folder (include extension)
# sample_video_name (if provided) should be the filename (without extension); it should be in the project paths "videos" folder, downloaded using video_downloader or uploaded (ensure .mp4 format if uploaded)
# NOTE: When passing in inference_video_folder or sample_video_name, it may be necessary to enclose it in quotation marks to avoid terminal marking it as another command

if torch.accelerator.is_available():
    device = torch.device(torch.accelerator.current_accelerator())
else:
    device = torch.device("cpu")

def main():

    if len(sys.argv) != 4:
        print("USAGE: python3 inference.py <model_name> <preexisting_video> [inference_video_name OR sample_video_name]")
        sys.exit(1)

    paths = ProjectPaths()
    model_name = sys.argv[1]
    inference_video_folder = os.path.join(paths.videos_for_inference_folder, sys.argv[3])

    if sys.argv[2] == "False":
        original_video_filename = f"{sys.argv[3]}.mp4"
        if not os.path.exists(inference_video_folder):
            os.mkdir(inference_video_folder)
        
        extract_clip_frames(video_path=os.path.join(paths.videos_folder, original_video_filename), output_dir=inference_video_folder)
        print("Frames extracted")
    elif sys.argv[2] == "True":
        pass
    else:
        print("arg <preexisting_video> must be either 'True' or 'False'")
        sys.exit(1)
    
    tracked_ball_points = obtain_tracked_ball_data(frame_folder_path=inference_video_folder, model_name=model_name)
    create_heatmap_video(inference_video_name=inference_video_folder)
    create_annotated_video(inference_video_name=inference_video_folder, tracked_ball_points=tracked_ball_points)

def load_model(model_name):
    paths = ProjectPaths()
    model_path = os.path.join(paths.final_model_save_folder, model_name)
    model = TrackNet()
    model_state = torch.load(model_path, weights_only=True, map_location=torch.device(device))
    model.load_state_dict(model_state)
    return model

def create_frame_sequence(i, seq_len=3):
    if i < seq_len - 1:
        return None
    frame_ids = list(range(i - seq_len + 1, i + 1))
    return frame_ids

def get_input(img_paths):
        imgs = []
        for img_path in img_paths:
            img = cv2.imread(img_path) # Keep in BGR format, the pretrained model was trained this way
            img = cv2.resize(img, (640, 360)) # Originally 1920, 1080
            imgs.append(img) # Each img of shape (h, w, c)
        
        imgs = np.concatenate(imgs, axis=2) # Concatenate along channel dimension; lower channel dimensions correspond to earlier frames
        imgs = np.moveaxis(imgs, 2, 0) # Shape now (c, h, w)
        imgs = imgs.astype(np.float32) / 255.0 # Normalize pixel values to [0, 1] as done in training
        return imgs

def filter_by_threshold(output, threshold=128):
    # For model output, map any pixels with value < t to 0, otherwise 255 (after using softmax) where t is threshold; in paper, t=128
    return np.where(output >= threshold, 255, 0).astype(np.uint8) # uint8 required for cv2.HoughCircles

def obtain_ball_center_coords(filtered_output):
    # Use Hough Gradient Method, and return the center coords of the ball; make sure only one circle is detected, otherwise no ball
    circles = cv2.HoughCircles( # Shape of circles is (1, n_circles_detected, 3); the first dim is byproduct of underlying implementation; the third consists of [x, y, radius]
        filtered_output,          # Your normalized, single-channel TrackNet heatmap
        method=cv2.HOUGH_GRADIENT,
        dp=1,                  # Keep at 1 for exact pixel accuracy
        minDist=15,            # Only 1 ball should be found
        param1=20,             # Low value to find the gradual intensity transitions, no sharp edges
        param2=10,             # Low value takens into account small/faint ball blobs when ball is far away
        minRadius=2,           # Accounts for the ball when it is furthest from camera
        maxRadius=30           # Accounts for the ball when it is closest to camera
    )

    if circles is None:
        return None
    
    if circles.shape[1] != 1:
        return None
    
    return list(circles[0][0])

def obtain_tracked_ball_data(frame_folder_path, model_name):
    model = load_model(model_name) # Load the trained model
    model.eval()

    # Loop through the frames and run each frame sequence through the model; frames should be sorted in order and have names of the form frame_########.jpg
    frames = [frame for frame in os.listdir(frame_folder_path) if not frame.startswith(".")] # All frames have a name of the form frame_########.jpg; ignore hidden files such as .DS_Store
    ball_points = {}
    # Make a directory to save the intermediate heatmaps before thresholding has been applied
    heatmap_folder = os.path.join(frame_folder_path, "heatmaps")
    if not os.path.exists(heatmap_folder):
        os.mkdir(heatmap_folder)
    with torch.inference_mode():
        for i in trange(len(frames), desc="Obtaining ball positions..."):
            frame_ids = create_frame_sequence(i)
            if frame_ids is None:
                heatmap = np.zeros((360, 640))# Write an empty intermediate heatmap
                cv2.imwrite(os.path.join(heatmap_folder, f"frame_{i:08d}_heatmap.png"), heatmap) # Save intermediate heatmaps
                continue
            frame_ids = list(map(lambda x: os.path.join(frame_folder_path, f"frame_{x:08d}.jpg"), frame_ids))
            model_input = np.expand_dims(get_input(frame_ids), axis=0) # Add batch dimension, batch size just equals 1
            model_input_tensor = torch.from_numpy(model_input).float().to(device)
            model_output_tensor = model(model_input_tensor, testing=True) # Shape is now (N, c, h, w) where N = 1
            model_output = model_output_tensor.detach().cpu().numpy()
            model_output = np.argmax(model_output, axis=1) # Shape is now (1, h, w); array of integers
            model_output = np.squeeze(model_output, axis=0) # Shape is now (h, w); 2d array required for cv2.HoughCircles
            cv2.imwrite(os.path.join(heatmap_folder, f"frame_{i:08d}_heatmap.png"), model_output) # Save intermediate heatmaps
            model_output = filter_by_threshold(model_output)
            ball_point = obtain_ball_center_coords(filtered_output=model_output)
            if ball_point is not None:
                scale_factor = 1080/360 # Multiply center coords by a scale factor to get from 360 x 640 to 1080 x 1920
                ball_point = list(map(lambda x: int(x*scale_factor), ball_point)) # ints used so it can be used when OpenCV draws circles
                ball_points[i] = ball_point
    return ball_points

def create_heatmap_video(inference_video_name):
    # Video saved to the same folder as the frames (not heatmaps)
    paths = ProjectPaths()
    output_path = os.path.join(paths.videos_for_inference_folder, inference_video_name, "heatmap_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Standard MP4 codec
    output_fps = 60.0
    frame_size = (1920, 1080)
    video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, frame_size)

    heatmap_folder_path = os.path.join(paths.videos_for_inference_folder, inference_video_name, "heatmaps") # Folder where saved heatmaps will be
    heatmaps = [heatmap for heatmap in os.listdir(heatmap_folder_path) if not heatmap.startswith(".")] # All heatmaps have a name of the form frame_########.png; ignore hidden files such as .DS_Store
    for i in trange(len(heatmaps), desc="Generating output heatmap video..."):
        heatmap_path = os.path.join(heatmap_folder_path, f"frame_{i:08d}_heatmap.png")
        heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
        heatmap = cv2.resize(heatmap, (1920, 1080))
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        video_writer.write(heatmap)
    video_writer.release()
    print("Heatmap video saved successfully!")

def create_annotated_video(inference_video_name, tracked_ball_points):
    # Video saved to the same folder as the frames
    paths = ProjectPaths()
    output_path = os.path.join(paths.videos_for_inference_folder, inference_video_name, "annotated_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Standard MP4 codec
    output_fps = 60.0
    frame_size = (1920, 1080)
    video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, frame_size)

    frame_folder_path = os.path.join(paths.videos_for_inference_folder, inference_video_name) # Folder where saved video frames will be
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