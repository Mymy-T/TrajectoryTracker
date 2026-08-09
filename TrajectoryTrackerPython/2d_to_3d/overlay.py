import cv2
import numpy as np

def overlay_trajectory_on_image(image_path, projected_trajectory, output_path=None, base_image=None):
    """
    - Draws projected_trajectory on top of the image at image_path and returns BGR image
    - If output_path is given, result is written to disk
    - If base_image is given (an already-loaded BGR ndarray), it's used instead of re-reading image_path from disk
    """
    if base_image is not None:
        img = base_image.copy()
    else:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None

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

    if output_path is not None:
        cv2.imwrite(output_path, img)
        print(f"Successfully generated overlay image saved to: {output_path}")

    return img