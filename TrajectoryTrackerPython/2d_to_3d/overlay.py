import cv2
import numpy as np

def overlay_trajectory_on_image(image_path, projected_trajectory, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    points = []
    for pt in projected_trajectory:
        pixel_x = int(round(pt[0]))
        pixel_y = int(round(pt[1])) 
        points.append((pixel_x, pixel_y))

    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)

    # Highlight endpoints
    cv2.circle(img, points[0], radius=8, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(img, points[-1], radius=8, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print(f"Successfully generated overlay image saved to: {output_path}")