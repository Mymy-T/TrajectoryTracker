import cv2
import math
import numpy as np

FADE_BANDS = 12 # number of alpha bands approximating a smooth fade within the trailing window; keeps the fade cost O(FADE_BANDS) instead of O(fade_window), since this runs every animation frame


def _draw_fading_trail(img, points, fade_window, color=(255, 0, 0), thickness=3):
    """
    Draws the trailing fade_window points as a comet trail: the newest segment is fully opaque,
    opacity ramps down linearly across the window, and points older than fade_window are omitted
    entirely. Composited in FADE_BANDS chunks rather than per-segment so cost stays bounded
    regardless of window size.
    """
    n_points = len(points)
    start_index = max(0, n_points - 1 - int(round(fade_window)))
    trail_points = points[start_index:]

    n_segments = len(trail_points) - 1
    if n_segments <= 0:
        return

    band_size = max(1, math.ceil(n_segments / FADE_BANDS))

    for band_start in range(0, n_segments, band_size):
        band_end = min(n_segments, band_start + band_size)
        band_points = np.array(trail_points[band_start:band_end + 1], dtype=np.int32)

        # age (in points) of this band's newer edge; 0 = the current/newest frame
        age = n_segments - band_end
        alpha = max(0.0, 1.0 - age / fade_window) if fade_window > 0 else 1.0

        overlay = img.copy()
        cv2.polylines(overlay, [band_points], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def overlay_trajectory_on_image(image_path, projected_trajectory, output_path=None, base_image=None, fade_window=None):
    """
    - Draws projected_trajectory on top of the image at image_path and returns BGR image
    - If output_path is given, result is written to disk
    - If base_image is given (an already-loaded BGR ndarray), it's used instead of re-reading image_path from disk
    - If fade_window is given (a number of trailing points), the trail is drawn as a comet trail:
      opacity ramps down to 0 over the trailing fade_window points and anything older is omitted
      entirely, instead of being drawn solid.
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

    if fade_window is not None and len(points) > 1:
        _draw_fading_trail(img, points, fade_window)
    else:
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)

    # Highlight endpoints
    cv2.circle(img, points[0], radius=8, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(img, points[-1], radius=8, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

    if output_path is not None:
        cv2.imwrite(output_path, img)
        print(f"Successfully generated overlay image saved to: {output_path}")

    return img