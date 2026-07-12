import cv2
import math
import numpy as np
import os
import pandas as pd
from pathlib import Path
from paths import ProjectPaths
import sys

data_df = pd.DataFrame({
        "path": [],
        "prev_path": [],
        "prev_prev_path": [],
        "heatmap_path": []
    })

def create_processed_dataset(data_path, processed_data_path):
    H = 1080
    W = 1920
    matches = [match for match in os.listdir(data_path) if not match.startswith(".")] # All matches have a name of the form match_####; ignore hidden files such as .DS_Store
    for match in matches:
        match_path = os.path.join(data_path, match)
        # Create output directory if it doesn't already exist
        out_match_folder = os.path.join(processed_data_path, match)
        if not os.path.exists(out_match_folder):
            os.mkdir(out_match_folder)
            print(f"{out_match_folder} created.")

        clips = [clip for clip in os.listdir(match_path) if not clip.startswith(".")] # All clips have a name of the form clip_####; ignore hidden files such as .DS_Store
        for clip in clips:
            clip_folder = os.path.join(match_path, clip)
            img_folder = os.path.join(clip_folder, "images")
            label_folder = os.path.join(clip_folder, "labels")
            img_names = sorted(os.listdir(img_folder)) # All files have a name of the form frame_########.jpg/png; rearrange alphabetically to match labels
            label_names = sorted(os.listdir(label_folder)) # All files have a name of the form frame_########.txt; rearrange alphabetically to match imgs
            classes_dict = {} # Obtain from the classes.txt file withing each clip folder; matches the label class_id to the appropriate classes (ball or negative in this project)
            with open(os.path.join(clip_folder, "classes.txt"), "r") as file:
                for line in file:
                    # Split line by spaces and unpack into key and values
                    parts = line.strip().split(" ")
                    key = int(parts[1]) # class_id
                    value = parts[0] # class name
                    classes_dict[key] = value

            # Test for folder existence
            out_clip_folder = os.path.join(out_match_folder, clip)
            if not os.path.exists(out_clip_folder):
                os.mkdir(out_clip_folder)
                print(f"{out_clip_folder} created.")
            out_img_folder = os.path.join(out_clip_folder, "images")
            out_heatmap_folder = os.path.join(out_clip_folder, "heatmaps")
            if not os.path.exists(out_img_folder):
                os.mkdir(out_img_folder)
                print(f"{out_img_folder} created.")
            if not os.path.exists(out_heatmap_folder):
                os.mkdir(out_heatmap_folder)
                print(f"{out_heatmap_folder} created.")
            num_already_existing_frames = 0
            for i in range(len(img_names)): # img_names and label_names have the same length
                # Check if the file already exists, and skip it if it does
                if os.path.exists(os.path.join(out_img_folder, img_names[i])):
                    num_already_existing_frames += 1
                    continue
                img = cv2.imread(os.path.join(img_folder, img_names[i]))
                with open(os.path.join(label_folder, label_names[i])) as label_file:
                    properties = label_file.read().split() # Properties will have the form [class_id, center_x, center_y, width, height], where coords and lengths are normalized
                    try:
                        class_id = int(properties[0])
                        center_x = float(properties[1]) * W
                        center_y = float(properties[2]) * H
                        w = float(properties[3])*W
                        h = float(properties[4])*H
                        s = (w + h) / 2.0 # Width & height should be the same
                        if classes_dict[class_id] == "negative":
                            negative = True
                        else:
                            negative = False
                        training_img, gt_heatmap = create_training_data(img, center_x, center_y, s, negative)
                        cv2.imwrite(os.path.join(out_img_folder, img_names[i]), training_img)
                        base_filename = Path(img_names[i]).stem
                        cv2.imwrite(os.path.join(out_heatmap_folder, f"{base_filename}_heatmap.png"), gt_heatmap)
                    except IndexError:
                        print(f"Index error: frame{i:04d}; see {os.path.join(label_folder, label_names[i])}. Frame skipped.")
            print(f"In {out_clip_folder}, {num_already_existing_frames} examples already existed;  {len(img_names) - num_already_existing_frames} examples were added, for a total of {len(img_names)} examples.")

    print("Training dataset created successfully!")



def create_training_data(img_1080p, old_x, old_y, old_s, negative): # All annotations are square bounding boxes, so h = w
    """
    Inputs - original image, coords of center of ball
    Outputs - resized image, gt heatmap
    """
    img_360p = cv2.resize(img_1080p, (640, 360))

    if negative == True:
        return img_360p, np.zeros((360, 640), dtype=np.uint8)
    
    # Scale label coords to match 640 by 360 img dims
    scale_factor = float(360/1080)
    x = int(round(old_x * scale_factor))
    y = int(round(old_y * scale_factor))
    s = int(round(old_s * scale_factor))

    if s <= 0:
        return img_360p, np.zeros((360, 640), dtype=np.uint8)
    
    heatmap_x = np.arange(0, 640, 1, float)
    heatmap_y= np.arange(0, 360, 1, float)
    heatmap_x, heatmap_y = np.meshgrid(heatmap_x, heatmap_y, indexing="xy")

    
    # Calculate sigma, see paper for details; the threshold is set to 73 (plug in r=5 and sigma^2 = 10 into original formula), so find sigma given that exp(-r^2 / 2*sigma^2) = 73/255 and that r = s/2
    # Rearranging and substituting r = s/2 gives sigma^2 = -s^2 / 8ln(73/255)
    sigma_sq = -(s**2) / (8*math.log(73/255))

    dist_sq = (heatmap_x - x)**2 + (heatmap_y - y)**2  # Utilize broadcasting
    gaussian = np.exp(-dist_sq / (2 * sigma_sq))
    heatmap_360p = (gaussian * 255).astype(np.uint8) # Final shape is (360 x 640), correct for cv2; values are confined between 0 and 255, inclusive
    
    return img_360p, heatmap_360p


if __name__ == "__main__":
    project_paths = ProjectPaths()
    create_processed_dataset(project_paths.annotated_data_folder, project_paths.processed_data_folder)