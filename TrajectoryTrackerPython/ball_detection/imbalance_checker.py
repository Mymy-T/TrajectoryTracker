import cv2
import numpy as np
import os
from paths import ProjectPaths

class_count_dict = dict([(c, 0) for c in range(256)]) # keys are classes (0-255), values are number of times each class appears
total_pixel_count = 0

processed_data_path = ProjectPaths().processed_data_folder

matches = sorted([match for match in os.listdir(processed_data_path) if not match.startswith(".")]) # All matches have a name of the form match_####; ignore hidden files such as .DS_Store
for match in matches:
    match_path = os.path.join(processed_data_path, match)
    clips = sorted([clip for clip in os.listdir(match_path) if not clip.startswith(".")]) # All clips have a name of the form clip_####; ignore hidden files such as .DS_Store
    for clip in clips:
        heatmaps_path = os.path.join(match_path, clip, "heatmaps")
        heatmaps = [heatmap for heatmap in os.listdir(heatmaps_path) if not heatmap.startswith(".")] # All heatmaps have a name of the form frame_########_heatmap.png; ignore hidden files such as .DS_Store
        for heatmap in heatmaps:
            heatmap_path = os.path.join(heatmaps_path, heatmap)
            img = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
            values, counts = np.unique(img, return_counts=True)
            temp_class_count_dict = dict(zip(values, counts))
            for k, v in temp_class_count_dict.items():
                class_count_dict[k] += v
                total_pixel_count += v

weights = np.array([min(total_pixel_count/count, 1000) if count != 0 else 0 for count in class_count_dict.values()])

def normalize_and_scale_weights(class_count_dict, total_px_count, clip_val):
    weights = []
    K = 0
    for count in class_count_dict.values():
        if count == 0:
            weights.append(0.0)
            K += 1
        else:
            weight = min(total_px_count/count, clip_val)
            weights.append(weight)
    
    weights = np.array(weights)
    total_weight = weights.sum()
    n_weights = weights/total_weight # Weights now sum to one, but avg of weights across active classes is 1/K, where K is the number of active classes
    n_weights = n_weights * K # Avg of weights across active classes is 1
    return n_weights


print(class_count_dict)
print(total_pixel_count)
print(weights)

weights = normalize_and_scale_weights(class_count_dict, total_pixel_count, 100)
print(weights)

# TODO: Figure out appropriate way to implement weighting