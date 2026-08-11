import cv2
from data_models import CourtProperties
import math
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

FADE_BANDS = 12 # number of alpha bands approximating a smooth fade within the trailing window; keeps the fade cost O(FADE_BANDS) instead of O(fade_window), since this runs every animation frame


def _draw_fading_trail(img, points, fade_window, color=(255, 0, 0), thickness=3):
    """
    Draws the trailing fade_window points as a comet trail: the newest segment is fully opaque, opacity decreases linearly across the window, and points older than fade_window are omitted entirely
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


def _plot_fading_line_2d(ax, x, y, fade_window, label=None, color=(0.0, 0.0, 1.0)):
    """
    The newest segment is fully opaque; opacity ramps down linearly over the trailing fade_window points, and anything older than fade_window points back is omitted entirely
    """
    n_points = len(x)
    start_index = max(0, n_points - 1 - int(round(fade_window)))

    xs = np.asarray(x[start_index:])
    ys = np.asarray(y[start_index:])
    if len(xs) < 2:
        return

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # age (in points) of each segment's newer endpoint; 0 = the current/newest frame
    newest_index = n_points - 1
    segment_end_indices = start_index + np.arange(1, len(segments) + 1)
    ages = newest_index - segment_end_indices
    alphas = np.clip(1.0 - ages / fade_window, 0.0, 1.0) if fade_window > 0 else np.ones(len(segments))

    colors = [(*color, a) for a in alphas]
    ax.add_collection(LineCollection(segments, colors=colors, linewidth=2, label=label))


def plot_2d_trajectory(trajectory, ax=None, fade_window=None):
    """
    If ax is None, creates a new figure/axes and calls plt.show() at the end (standalone use);
    else, draws onto it and leaves creating/showing the figure to the caller (eg for embedding in GUI)
    """
    court_properties = CourtProperties()
    l = court_properties.court_length
    w = court_properties.court_width

    # t = np.linspace(0, end_time, int(10/0.001))
    x = [point[0] for point in trajectory]
    y = [1080 - point[1] for point in trajectory]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    if fade_window is not None and len(x) > 1:
        _plot_fading_line_2d(ax, x, y, fade_window, label="Projected 2D Trajectory")
    else:
        ax.plot(x, y, label="Projected 2D Trajectory", color="b", linewidth=2)

    # Highlight start and end points
    ax.scatter(x[0], y[0], color="green", s=50, label="Start")
    ax.scatter(x[-1], y[-1], color="red", s=50, label="End")

    # Set limits based on court dimensions or pixel limits if applicable
    # TODO: convert to 640 by 360 to simulate true data
    ax.set_xlim([0, 1920])
    ax.set_ylim([0, 1080])

    ax.set_xlabel("X Axis (2D Projection)")
    ax.set_ylabel("Y Axis (2D Projection)")
    ax.set_title("2D Projected Trajectory")
    ax.legend()

    if standalone:
        plt.show()


def _plot_fading_line_3d(ax, plot_x, plot_y, plot_z, fade_window, label=None, color=(0.0, 0.0, 1.0)):
    """
    The newest segment is fully opaque; opacity ramps down linearly over the trailing fade_window points, and anything older than fade_window points back is omitted entirely
    """
    n_points = len(plot_x)
    start_index = max(0, n_points - 1 - int(round(fade_window)))

    x = np.asarray(plot_x[start_index:])
    y = np.asarray(plot_y[start_index:])
    z = np.asarray(plot_z[start_index:])
    if len(x) < 2:
        return

    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # age (in points) of each segment's newer endpoint; 0 = the current/newest frame
    newest_index = n_points - 1
    segment_end_indices = start_index + np.arange(1, len(segments) + 1)
    ages = newest_index - segment_end_indices
    alphas = np.clip(1.0 - ages / fade_window, 0.0, 1.0) if fade_window > 0 else np.ones(len(segments))

    colors = [(*color, a) for a in alphas]
    ax.add_collection3d(Line3DCollection(segments, colors=colors, linewidth=2, label=label))


def plot_3d_trajectory(trajectory, ax=None, fade_window=None):
    """
    If ax is None, creates a new figure/3D axes and calls plt.show() at the end (standalone use);
    else, draws onto it and leaves creating/showing the figure to the caller (eg for embedding in GUI)
    """
    court_properties = CourtProperties()
    l = court_properties.court_length
    w = court_properties.court_width
    sw = court_properties.singles_width
    sld = court_properties.service_line_dist_from_net

    x = []
    y = []
    z = []
    for point in trajectory:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])

    # Plotting
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")


    # Outer court lines
    ax.plot(
        [-l/2, l/2, l/2, -l/2, -l/2],
        [-w/2, -w/2, w/2, w/2, -w/2],
        [0, 0, 0, 0, 0],
        color="green",
        linewidth=1,
        label="Court Lines"
    )

    # Singles lines
    ax.plot(
        [-l/2, l/2],
        [-sw/2, -sw/2],
        [0, 0],
        color="green",
        linewidth=1,
        label="Court Lines"
    )

    ax.plot(
        [-l/2, l/2],
        [sw/2, sw/2],
        [0, 0],
        color="green",
        linewidth=1,
        label="Court Lines"
    )

    # Net line (across X = 0)
    ax.plot([0, 0], [-w/2, w/2], [0, 0], color="green", linewidth=1.5)

    # Service lines
    ax.plot(
        [-sld, -sld],
        [-w/2, w/2],
        [0, 0],
        color="green",
        linewidth=1,
    )
    ax.plot(
        [sld, sld],
        [-w/2, w/2],
        [0, 0],
        color="green",
        linewidth=1,
    )

    # Center Service Line (Connects the two service lines down the middle at Z = 0)
    ax.plot(
        [-sld, sld],
        [0, 0],
        [0, 0],
        color="green",
        linewidth=1,
    )


    # Plot trajectory
    if fade_window is not None and len(x) > 1:
        _plot_fading_line_3d(ax, x, z, y, fade_window, label="3D Trajectory")
    else:
        ax.plot(x, z, y, label="3D Trajectory", color="b", linewidth=2)

    # Highlight start and end points
    ax.scatter(x[0], z[0], y[0], color="green", s=50, label="Start")
    ax.scatter(x[-1], z[-1], y[-1], color="red", s=50, label="End")

    ax.set_xlim([-l/2 - 1, l/2 + 1])
    ax.set_ylim([-w/2 - 1, w/2 + 1])
    ax.invert_yaxis() # This way negative values of z axis are towards the left
    ax.set_zlim([0, 4])
    ax.set_box_aspect([l, w, 4])

    # NOTE: In matplotlib, +z is up; in the coordinates of the trajectory, +y is up
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Z Axis")
    ax.set_zlabel("Y Axis")
    ax.set_title("3D Path Trajectory")
    ax.grid(False)
    ax.legend()

    if standalone:
        plt.show()
