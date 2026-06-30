import cv2
import math
import numpy as np
import os
from pathlib import Path
from paths import ProjectPaths



def create_training_dataset(data_path, training_data_path):
    H = 1080
    W = 1920
    clips = [clip for clip in os.listdir(data_path) if not clip.startswith('.')] # All clips have a name of the form clip_####; ignore hidden files such as .DS_Store
    for clip in clips:
        clip_folder = os.path.join(data_path, clip)
        img_folder = os.path.join(clip_folder, "images")
        label_folder = os.path.join(clip_folder, "labels")
        img_names = sorted(os.listdir(img_folder)) # All files have a name of the form frame_########.jpg/png; rearrange alphabetically to match labels
        label_names = sorted(os.listdir(label_folder)) # All files have a name of the form frame_########.txt; rearrange alphabetically to match imgs
        try: # Create output directory; this should not exist yet!
            out_clip_folder = os.path.join(training_data_path, clip)
            os.mkdir(out_clip_folder)
            out_img_folder = os.path.join(out_clip_folder, "images")
            out_heatmap_folder = os.path.join(out_clip_folder, "heatmaps")
            os.mkdir(out_img_folder)
            os.mkdir(out_heatmap_folder)
        except FileExistsError:
            print(f"Make sure {training_data_path} is empty before running this function.")
            raise

        for i in range(len(img_names)): # img_names and label_names have the same length
            img = cv2.imread(os.path.join(img_folder, img_names[i]))
            with open(os.path.join(label_folder, label_names[i])) as label_file:
                properties = label_file.read().split() # Properties will have the form [class_id, center_x, center_y, width, height], where coords and lengths are normalized
                center_x = float(properties[1]) * W
                center_y = float(properties[2]) * H
                w = float(properties[3])*W
                h = float(properties[4])*H
                s = (w + h) / 2.0 # Width & height should be the same
            training_img, gt_heatmap = create_training_data(img, center_x, center_y, s)
            cv2.imwrite(os.path.join(out_img_folder, img_names[i]), training_img)
            base_filename = Path(img_names[i]).stem
            cv2.imwrite(os.path.join(out_heatmap_folder, f"{base_filename}_heatmap.png"), gt_heatmap)
    print("Training dataset created successfully!")



def create_training_data(img_1080p, old_x, old_y, old_s): # All annotations are square bounding boxes, so h = w
    """
    Inputs - original image, coords of center of ball
    Outputs - resized image, gt heatmap
    """
    img_360p = cv2.resize(img_1080p, (640, 360))
    scale_factor = float(360/1080)
    
    # Scale label coords to match 640 by 360 img dims
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
    heatmap_360p = (gaussian * 255).astype(np.uint8) # Final shape is (360 x 640), correct for cv2
    
    return img_360p, heatmap_360p


if __name__ == "__main__":
    project_paths = ProjectPaths()
    create_training_dataset(project_paths.data_folder, project_paths.train_data_folder)