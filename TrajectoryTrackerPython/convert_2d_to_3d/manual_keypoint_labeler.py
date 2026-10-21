from convert_2d_to_3d.data_models import ADDITIONAL_CALIBRATION_KEYPOINT_NAMES, BASELINE_CORNER_KEYPOINT_NAMES, SERVICE_LINE_CORNER_KEYPOINT_NAMES
from convert_2d_to_3d.paths import Paths
from convert_2d_to_3d import settings
import cv2
import numpy as np
import os


WINDOW_NAME = "Camera Calibration - Keypoint Labeler"

_PLACED_COLOUR = (0, 0, 255)
_NEXT_COLOUR = (0, 200, 255)
_TEXT_COLOUR = (255, 255, 255)
_BANNER_COLOUR = (0, 0, 0)


def add_point(points, x, y, required_num_points):
    """
    Returns a new list with (x, y) appended, or the list unchanged if all points are already placed.
    Kept separate from the GUI so the click/undo logic can be tested headlessly.
    """
    if len(points) >= required_num_points:
        return list(points)

    return list(points) + [np.array([x, y], dtype=np.float32)]


def undo_point(points):
    return list(points[:-1])


def _keypoint_names():
    """
    The click order, which must match the world coord order used in projector.py.

    Which four corners are wanted depends on settings.corner_keypoint_set, matching
    projector._generate_3d_baseline_coords. 4 points are needed when the intrinsics are already
    known (PnP only), 9 when they must be solved for too.
    """
    data_gen_settings = settings.SyntheticDataGeneration()

    if data_gen_settings.corner_keypoint_set == settings.WorldCoordSet.SERVICE_LINE:
        corner_names = SERVICE_LINE_CORNER_KEYPOINT_NAMES
    else:
        corner_names = BASELINE_CORNER_KEYPOINT_NAMES

    if data_gen_settings.intrinsics_known:
        return corner_names

    return corner_names + ADDITIONAL_CALIBRATION_KEYPOINT_NAMES


def _load_reference_image(image_path):
    if not os.path.exists(image_path):
        raise RuntimeError(f"Make sure an image exists at {image_path}; this is used as the reference image for selecting keypoints.")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not open or decode the image at {image_path}.")

    height, width = img.shape[0:2]
    expected_image_size = settings.SyntheticDataGeneration().expected_image_size
    if (width, height) != tuple(expected_image_size):
        expected_width, expected_height = expected_image_size
        raise RuntimeError(
            f"Calibration image at {image_path} is {width}x{height}, but {expected_width}x{expected_height} is required. "
            "The principal point is fixed at the image centre during calibration, so a different resolution silently biases the result. "
            "Re-download the clip with ball_detection's video downloader, which enforces this resolution."
        )

    return img


def _draw_banner(canvas, lines):
    """
    Draws the instruction text over a dark strip at the top so it stays readable on any footage.
    """
    banner_height = 26 * len(lines) + 14
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], banner_height), _BANNER_COLOUR, -1)

    for i, (text, colour) in enumerate(lines):
        cv2.putText(canvas, text, (12, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)


def _render(base_image, points, names, subtitle=""):
    """
    Redraws from a clean copy each frame, so undoing a point actually removes it from the display
    """
    canvas = base_image.copy()

    for i, point in enumerate(points):
        centre = (int(round(float(point[0]))), int(round(float(point[1]))))
        cv2.circle(canvas, centre, 6, _PLACED_COLOUR, -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, str(i + 1), (centre[0] + 10, centre[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _PLACED_COLOUR, 2, cv2.LINE_AA)

    if len(points) < len(names):
        lines = [
            (f"Point {len(points) + 1} of {len(names)}: {names[len(points)]}", _NEXT_COLOUR),
            ("Left click to place  |  right click or 'u' to undo  |  'q'/esc to cancel", _TEXT_COLOUR)
        ]
    else:
        lines = [
            (f"All {len(names)} points placed.", _NEXT_COLOUR),
            ("Press enter to confirm and save  |  right click or 'u' to undo  |  'q'/esc to cancel", _TEXT_COLOUR)
        ]

    if subtitle:
        lines.append((subtitle, _TEXT_COLOUR))

    _draw_banner(canvas, lines)
    return canvas


def obtain_keypoints(verbose=False):
    """
    Prompts for each calibration keypoint in turn and saves them, in the order projector.py expects.

    Left click places the next point, right click (or 'u') undoes the last one, and enter confirms
    once every point is placed. Nothing is saved unless the user confirms.
    """

    paths = Paths()
    data_gen_settings = settings.SyntheticDataGeneration()
    names = _keypoint_names()
    required_num_points = len(names)

    subtitle = f"{data_gen_settings.camera_perspective_name}  |  corner set: {data_gen_settings.corner_keypoint_set.name}  |  left/right as seen here, camera behind the NEAR baseline"

    base_image = _load_reference_image(paths.sample_image_path)

    state = {"points": []}

    def _click_event(event, x, y, flags, params): # Callback function
        if event == cv2.EVENT_LBUTTONDOWN:
            state["points"] = add_point(state["points"], x, y, required_num_points)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["points"] = undo_point(state["points"])

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _click_event)

    print(f"Select {required_num_points} keypoints in the order prompted on screen.")
    print("Left click to place, right click or 'u' to undo, enter to confirm, 'q'/esc to cancel.")

    confirmed = False
    try:
        while True:
            cv2.imshow(WINDOW_NAME, _render(base_image, state["points"], names, subtitle))
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q") or key == 27:
                break

            if key == ord("u"):
                state["points"] = undo_point(state["points"])

            if key in (13, 10) and len(state["points"]) == required_num_points: # Enter
                confirmed = True
                break
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1) # Lets the window actually close on macOS

    if not confirmed:
        raise RuntimeError(f"Keypoint labeling cancelled; {len(state['points'])} of {required_num_points} points were placed and nothing was saved.")

    points = np.stack(state["points"], axis=0)
    np.save(paths.saved_keypoints_path, points)

    if verbose:
        print("--- Final Selected Coordinates ---")
        for name, point in zip(names, points):
            print(f"  {name}: ({point[0]:.1f}, {point[1]:.1f})")

    return points


if __name__ == "__main__":
    obtain_keypoints(verbose=True)
