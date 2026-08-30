import cv2
import numpy as np
from paths import Paths


def obtain_keypoints(verbose=False):

    def _click_event(event, x, y, flags, params): # Callback function
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.append(np.array([x, y], dtype=np.float32))
            
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image Viewer", img)

    selected_points = []

    paths = Paths()
    image_path = paths.sample_image_path

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not open or find the image.")
        exit()

    cv2.namedWindow("Image Viewer")
    cv2.setMouseCallback("Image Viewer", _click_event)

    print("Click on the image to select points. Press 'q' or 'esc' to exit and view final list.")
    while True:
        cv2.imshow("Image Viewer", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()

    if len(selected_points) == 4:
        points = np.stack(selected_points, axis=0)
        np.save(paths.saved_keypoints_path, points)
    else:
        print("Error: Must select 4 points")
        exit()

    if verbose:
        print("--- Final Selected Coordinates ---")
        print(selected_points)
