import cv2
import numpy as np


def create_training_img(img_1080p, original_x_label, original_y_label):
    """
    Inputs - original image, coords of center of ball
    Outputs - resized image, gt heatmap
    """
    img_360p = cv2.resize(img_1080p, (640, 360))
    
    x_label = int(round(x / 3.0))
    y_label = int(round(y / 3.0))
    
    x = np.arange(0, 640, 1, float)
    y = np.arange(0, 360, 1, float)
    x, y = np.meshgrid(x, y, indexing="ij")
    
    dist_sq = (x - x_label)**2 + (y - y_label)**2  # utilize broadcasting
    gaussian = np.exp(-dist_sq / (2 * 10)) # sigma^2 = 10, see paper
    heatmap_360p = (gaussian * 255).astype(np.uint8)
    
    return img_360p, heatmap_360p
