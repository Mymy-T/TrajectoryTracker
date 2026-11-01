from convert_2d_to_3d.data_models import CourtProperties
from convert_2d_to_3d.paths import Paths
from convert_2d_to_3d.physics import generate_sample_trajectory
from convert_2d_to_3d import projector as proj
from convert_2d_to_3d import settings
from convert_2d_to_3d.visualizer import overlay_trajectory_on_image
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import tkinter as tk
from tkinter import ttk

DISPLAY_IMAGE_SIZE = (1280, 720) # (width, height); the overlay is drawn at full resolution, then downsized to this for display
SLIDER_THROTTLE_MS = 40 # cap redraw rate while dragging a slider

LINE_SAMPLES_PER_SEGMENT = 50 # each court line is sampled densely so it can be clipped where it passes behind the camera
MIN_DEPTH = 1e-3 # points closer than this (meters, camera frame z) are treated as behind the camera

FOCAL_RANGE_FRACTION = 0.5 # fx/fy sliders span base * (1 +/- this)
POSITION_RANGE = 15.0 # meters either side of the base camera position
ROTATION_RANGE_DEG = 30.0

# Step buttons shown either side of each slider, for nudges finer than a slider drag allows
INTRINSICS_STEPS_PX = (10, 50, 100)
POSITION_STEPS_M = (0.1, 0.5, 1.0) # 0.1 m is on the order of the translation jitter; 1 m is a coarse reposition
DEFAULT_NUM_ERROR_SAMPLES = 100 # camera configs sampled by "Sample error stats"

ROTATION_STEPS_DEG = (0.1, 0.5, 1.0) # At broadcast focal lengths (~2900 px), 0.1 deg ~ 5 px and 1 deg ~ 50 px of image shift

_COURT_COLOUR = (0, 255, 255)
_BASE_COURT_COLOUR = (160, 160, 160)
_NET_COLOUR = (255, 0, 255)
_LABELLED_KEYPOINT_COLOUR = (0, 0, 255)
_PROJECTED_KEYPOINT_COLOUR = (0, 255, 0)


def _camera_position_from_extrinsics(extrinsics: np.ndarray):
    r_matrix = extrinsics[0:3, 0:3]
    t_vec = extrinsics[0:3, 3]
    return -r_matrix.T @ t_vec # t = -RT -> T = -R^-1 @ t = -R^T @ t


def _court_line_segments():
    """
    Returns (court_segments, net_segments); each is a list of (start, end) pairs of 3D world coords
    - x runs along the court length (near baseline at -l/2), y is up, z runs across the court (left sideline at -w/2), matching projector.py
    """
    court_properties = CourtProperties()
    l = court_properties.court_length
    w = court_properties.court_width
    sw = court_properties.singles_width
    sld = court_properties.service_line_dist_from_net
    nph = court_properties.net_height_at_posts
    nch = court_properties.net_height_at_center
    npdfs = court_properties.net_post_distance_from_sideline

    court_segments = [
        ((-l/2, 0, -w/2), (l/2, 0, -w/2)), # Doubles sidelines
        ((-l/2, 0, w/2), (l/2, 0, w/2)),
        ((-l/2, 0, -sw/2), (l/2, 0, -sw/2)), # Singles sidelines
        ((-l/2, 0, sw/2), (l/2, 0, sw/2)),
        ((-l/2, 0, -w/2), (-l/2, 0, w/2)), # Baselines
        ((l/2, 0, -w/2), (l/2, 0, w/2)),
        ((-sld, 0, -sw/2), (-sld, 0, sw/2)), # Service lines
        ((sld, 0, -sw/2), (sld, 0, sw/2)),
        ((-sld, 0, 0), (sld, 0, 0)) # Centre service line
    ]

    # Net post placement mirrors projector._generate_full_calibration_3d_coords
    if settings.SyntheticDataGeneration().net_post_type == settings.NetPostType.DOUBLES:
        post_z = w/2 + npdfs
    else:
        post_z = sw/2 + npdfs

    net_segments = [
        ((0, 0, -post_z), (0, 0, post_z)), # Base of net
        ((0, nph, -post_z), (0, nch, 0)), # Top of net, sagging to centre height
        ((0, nch, 0), (0, nph, post_z)),
        ((0, 0, -post_z), (0, nph, -post_z)), # Posts
        ((0, 0, post_z), (0, nph, post_z))
    ]

    to_array = lambda segments: [(np.array(start, dtype=np.float64), np.array(end, dtype=np.float64)) for start, end in segments]
    return to_array(court_segments), to_array(net_segments)


def _project_with_depth(points: np.ndarray, projection_matrix: np.ndarray):
    """
    Returns (pixels, depths) for an (N, 3) array of world points
    - The third homogeneous coord of K[I|0]E @ X is the camera frame z, since the last row of K is [0, 0, 1]
    """
    homogeneous_projection = (projection_matrix @ proj.homogenize(points).T).T # (N, 3)
    depths = homogeneous_projection[:, 2]
    pixels = proj.dehomogenize(homogeneous_projection) # (N, 2)
    return pixels, depths


def _draw_segment(img, start, end, projection_matrix, colour, thickness):
    """
    Projects a straight 3D segment and draws only the parts in front of the camera
    """
    t = np.linspace(0.0, 1.0, LINE_SAMPLES_PER_SEGMENT)[:, None]
    points = start + t * (end - start) # (N, 3)
    pixels, depths = _project_with_depth(points, projection_matrix)

    in_front = depths > MIN_DEPTH
    pixels = np.clip(pixels, -1e5, 1e5) # Keeps near-horizon points within int32 range for cv2

    # Split into contiguous runs of in-front points so lines never wrap through the vanishing point
    run = []
    for pixel, visible in zip(pixels, in_front):
        if visible:
            run.append(pixel)
        elif run:
            if len(run) > 1:
                cv2.polylines(img, [np.round(np.array(run)).astype(np.int32)], False, colour, thickness, cv2.LINE_AA)
            run = []
    if len(run) > 1:
        cv2.polylines(img, [np.round(np.array(run)).astype(np.int32)], False, colour, thickness, cv2.LINE_AA)


def _rms_reprojection_error(labelled_points: np.ndarray, projected_points: np.ndarray):
    return float(np.sqrt(np.mean(np.sum((labelled_points - projected_points) ** 2, axis=1))))


class CamConfigExplorerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera Config Explorer")
        self.geometry("1800x950")

        self.paths = Paths()
        self.data_gen_settings = settings.SyntheticDataGeneration()

        self.base_intrinsics, self.base_extrinsics = proj.get_base_camera_config()
        self.base_intrinsics = self.base_intrinsics.astype(np.float64)
        self.base_extrinsics = self.base_extrinsics.astype(np.float64)
        self.base_r_matrix = self.base_extrinsics[0:3, 0:3]
        self.base_camera_position = _camera_position_from_extrinsics(self.base_extrinsics)
        self.base_projection_matrix = proj.create_projection_matrix(self.base_intrinsics, self.base_extrinsics)

        full_image = cv2.imread(self.paths.sample_image_path)
        if full_image is None:
            raise RuntimeError(f"Could not open the sample image at {self.paths.sample_image_path}.")
        self._base_image = full_image

        # Labelled keypoints are 4 points if intrinsics were known (PnP), 9 if a full calibration was done; see manual_keypoint_labeler._keypoint_names
        self.labelled_keypoints = np.load(self.paths.saved_keypoints_path).astype(np.float64)
        if len(self.labelled_keypoints) == 4:
            self.keypoint_world_coords = proj._generate_3d_baseline_coords(self.data_gen_settings.corner_keypoint_set).astype(np.float64)
        else:
            self.keypoint_world_coords = proj._generate_full_calibration_3d_coords(self.data_gen_settings.corner_keypoint_set).astype(np.float64)

        self.court_segments, self.net_segments = _court_line_segments()

        self._sample_trajectory = None # Generated lazily the first time "Show sample trajectory" is ticked
        self._rng = np.random.default_rng()
        self._pending_render_after_id = None

        self.param_vars = {}
        self.value_labels = {}
        self.param_ranges = {}
        self.lock_aspect = tk.BooleanVar(value=True) # Full calibration uses CALIB_FIX_ASPECT_RATIO, so fx = fy by default
        self.show_base = tk.BooleanVar(value=False)
        self.show_keypoints = tk.BooleanVar(value=True)
        self.show_trajectory = tk.BooleanVar(value=False)
        self.num_error_samples = tk.IntVar(value=DEFAULT_NUM_ERROR_SAMPLES)

        self._build_widgets()
        self._reset_to_base()

    def _param_specs(self):
        """
        Returns a list of (section, key, label, from, to, format, steps) for every slider
        """
        fx = self.base_intrinsics[0, 0]
        fy = self.base_intrinsics[1, 1]
        width, height = self.data_gen_settings.expected_image_size
        X, Y, Z = self.base_camera_position

        return [
            ("Intrinsics (px)", "fx", "fx", fx * (1 - FOCAL_RANGE_FRACTION), fx * (1 + FOCAL_RANGE_FRACTION), "{:.1f}", INTRINSICS_STEPS_PX),
            ("Intrinsics (px)", "fy", "fy", fy * (1 - FOCAL_RANGE_FRACTION), fy * (1 + FOCAL_RANGE_FRACTION), "{:.1f}", INTRINSICS_STEPS_PX),
            ("Intrinsics (px)", "cx", "cx", 0.0, float(width), "{:.1f}", INTRINSICS_STEPS_PX),
            ("Intrinsics (px)", "cy", "cy", 0.0, float(height), "{:.1f}", INTRINSICS_STEPS_PX),
            ("Camera position, world coords (m)", "X", "X (along court)", X - POSITION_RANGE, X + POSITION_RANGE, "{:.2f}", POSITION_STEPS_M),
            ("Camera position, world coords (m)", "Y", "Y (height)", max(0.1, Y - POSITION_RANGE), Y + POSITION_RANGE, "{:.2f}", POSITION_STEPS_M),
            ("Camera position, world coords (m)", "Z", "Z (across court)", Z - POSITION_RANGE, Z + POSITION_RANGE, "{:.2f}", POSITION_STEPS_M),
            ("Rotation from base, camera frame (deg)", "tilt", "tilt (about cam x)", -ROTATION_RANGE_DEG, ROTATION_RANGE_DEG, "{:.2f}", ROTATION_STEPS_DEG),
            ("Rotation from base, camera frame (deg)", "pan", "pan (about cam y)", -ROTATION_RANGE_DEG, ROTATION_RANGE_DEG, "{:.2f}", ROTATION_STEPS_DEG),
            ("Rotation from base, camera frame (deg)", "roll", "roll (about cam z)", -ROTATION_RANGE_DEG, ROTATION_RANGE_DEG, "{:.2f}", ROTATION_STEPS_DEG)
        ]

    def _build_widgets(self):
        # Left: overlaid sample image
        image_frame = ttk.Frame(self)

        self.fig = Figure(figsize=(12.8, 7.2))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self._image_artist = self.ax.imshow(np.zeros((DISPLAY_IMAGE_SIZE[1], DISPLAY_IMAGE_SIZE[0], 3), dtype=np.uint8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right: sliders, toggles, buttons, info
        controls_frame = ttk.Frame(self)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8) # Packed after the controls so the image shrinks first when the window is narrow

        ttk.Style(self).configure("Step.TButton", padding=0, font=("TkDefaultFont", 9))

        ttk.Label(controls_frame, text=f"Perspective: {self.data_gen_settings.camera_perspective_name}", foreground="blue").pack(side=tk.TOP, anchor=tk.W)

        current_section = None
        for section, key, label, from_, to, fmt, steps in self._param_specs():
            if section != current_section:
                ttk.Label(controls_frame, text=section, font=("TkDefaultFont", 12, "bold")).pack(side=tk.TOP, anchor=tk.W, pady=(10, 2))
                current_section = section

            row = ttk.Frame(controls_frame)
            row.pack(side=tk.TOP, fill=tk.X)

            ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)

            var = tk.DoubleVar(value=from_)
            slider = ttk.Scale(row, from_=from_, to=to, orient=tk.HORIZONTAL, variable=var, length=220,
                               command=lambda _value, key=key: self._on_slider_change(key))
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))

            value_label = ttk.Label(row, text="", width=9)
            value_label.pack(side=tk.LEFT)

            # Step buttons go on their own row under the slider; aqua buttons are too wide to fit six beside it
            step_row = ttk.Frame(controls_frame)
            step_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

            ttk.Label(step_row, text="", width=18).pack(side=tk.LEFT) # Aligns the buttons under the slider
            ttk.Label(step_row, text="", width=9).pack(side=tk.RIGHT)

            for step in reversed(steps):
                ttk.Button(step_row, text=f"{-step:+g}", width=4, style="Step.TButton",
                           command=lambda key=key, step=step: self._on_step(key, -step)).pack(side=tk.LEFT)

            for step in reversed(steps): # Packing to the right reverses the order, so this reads +small ... +large
                ttk.Button(step_row, text=f"{step:+g}", width=4, style="Step.TButton",
                           command=lambda key=key, step=step: self._on_step(key, step)).pack(side=tk.RIGHT)

            self.param_vars[key] = var
            self.value_labels[key] = (value_label, fmt)
            self.param_ranges[key] = (from_, to)

        toggles_frame = ttk.Frame(controls_frame)
        toggles_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(toggles_frame, text="Lock fx = fy", variable=self.lock_aspect, command=self._on_lock_aspect_toggle).pack(side=tk.TOP, anchor=tk.W)
        ttk.Checkbutton(toggles_frame, text="Show base config wireframe", variable=self.show_base, command=self._schedule_render).pack(side=tk.TOP, anchor=tk.W)
        ttk.Checkbutton(toggles_frame, text="Show keypoints (red = labelled, green = projected)", variable=self.show_keypoints, command=self._schedule_render).pack(side=tk.TOP, anchor=tk.W)
        ttk.Checkbutton(toggles_frame, text="Show sample trajectory", variable=self.show_trajectory, command=self._schedule_render).pack(side=tk.TOP, anchor=tk.W)

        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        ttk.Button(buttons_frame, text="Reset to base", command=self._reset_to_base).pack(side=tk.LEFT)
        ttk.Button(buttons_frame, text="Sample jitter", command=self._sample_jitter).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(buttons_frame, text="Print config", command=self._print_config).pack(side=tk.LEFT, padx=(4, 0))

        error_stats_frame = ttk.Frame(controls_frame)
        error_stats_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        ttk.Label(error_stats_frame, text="Configs:").pack(side=tk.LEFT)
        ttk.Spinbox(error_stats_frame, from_=1, to=100000, increment=50, width=7, textvariable=self.num_error_samples).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(error_stats_frame, text="Sample error stats", command=self._sample_error_stats).pack(side=tk.LEFT)

        self.error_stats_label = ttk.Label(controls_frame, text="", justify=tk.LEFT, font=("TkFixedFont", 11))
        self.error_stats_label.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        self.info_label = ttk.Label(controls_frame, text="", justify=tk.LEFT, font=("TkFixedFont", 11))
        self.info_label.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))

    # MARK: Slider State
    def _set_params_from_config(self, intrinsics: np.ndarray, extrinsics: np.ndarray):
        """
        Loads an arbitrary (intrinsics, extrinsics) pair into the sliders; the inverse of _current_config
        """
        r_matrix = extrinsics[0:3, 0:3].astype(np.float64)
        camera_position = _camera_position_from_extrinsics(extrinsics.astype(np.float64))

        r_delta = r_matrix @ self.base_r_matrix.T # R = R_delta @ R_base -> R_delta = R @ R_base^T
        r_vec, _ = cv2.Rodrigues(r_delta)
        tilt, pan, roll = np.rad2deg(r_vec.flatten())

        values = {
            "fx": intrinsics[0, 0], "fy": intrinsics[1, 1], "cx": intrinsics[0, 2], "cy": intrinsics[1, 2],
            "X": camera_position[0], "Y": camera_position[1], "Z": camera_position[2],
            "tilt": tilt, "pan": pan, "roll": roll
        }

        if self.lock_aspect.get() and not np.isclose(values["fx"], values["fy"]):
            self.lock_aspect.set(False) # Jittered configs have independent fx and fy

        for key, value in values.items():
            self.param_vars[key].set(float(value))

        self._schedule_render()

    def _current_config(self):
        """
        Returns (intrinsics, extrinsics) built from the slider state
        - The rotation delta is composed on top of the base rotation in the camera frame, the same convention as projector.sample_camera_config
        - Camera position is held fixed while rotating (t is recomputed from it), so the rotation sliders behave like turning the camera on a tripod
        """
        p = {key: var.get() for key, var in self.param_vars.items()}

        intrinsics = np.array([[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]], dtype=np.float64)

        r_vec = np.deg2rad(np.array([p["tilt"], p["pan"], p["roll"]], dtype=np.float64))
        r_delta, _ = cv2.Rodrigues(r_vec)
        r_matrix = r_delta @ self.base_r_matrix

        camera_position = np.array([p["X"], p["Y"], p["Z"]], dtype=np.float64)

        extrinsics = np.identity(4, dtype=np.float64)
        extrinsics[0:3, 0:3] = r_matrix
        extrinsics[0:3, 3] = -r_matrix @ camera_position # t = -RT

        return intrinsics, extrinsics

    def _on_slider_change(self, key):
        if self.lock_aspect.get() and key in ("fx", "fy"):
            other = "fy" if key == "fx" else "fx"
            self.param_vars[other].set(self.param_vars[key].get()) # Setting a var directly doesn't fire its slider's command, so this can't recurse

        self._schedule_render()

    def _on_step(self, key, step):
        """
        Nudges a parameter by step, clamped to its slider's range so the slider and the value never disagree
        """
        from_, to = self.param_ranges[key]
        self.param_vars[key].set(float(np.clip(self.param_vars[key].get() + step, from_, to)))
        self._on_slider_change(key) # Keeps fx = fy in sync when locked

    def _on_lock_aspect_toggle(self):
        if self.lock_aspect.get():
            self.param_vars["fy"].set(self.param_vars["fx"].get())
        self._schedule_render()

    def _reset_to_base(self):
        self.lock_aspect.set(bool(np.isclose(self.base_intrinsics[0, 0], self.base_intrinsics[1, 1])))
        self._set_params_from_config(self.base_intrinsics, self.base_extrinsics)

    def _sample_jitter(self):
        intrinsics, extrinsics = proj.sample_camera_config(self._rng, self.base_intrinsics, self.base_extrinsics, self.data_gen_settings.camera_config_sampling_settings)
        self._set_params_from_config(intrinsics, extrinsics)

    def _sample_error_stats(self):
        """
        Samples camera configs the same way dataset creation does (base config + settings' jitter) and reports the keypoint RMS reprojection error across them
        - This is how far the labelled keypoints are from where each sampled camera says they should be, i.e. how much the jitter moves the court in the image
        """
        try:
            num_samples = int(self.num_error_samples.get())
        except (tk.TclError, ValueError):
            self.error_stats_label.config(text="Number of configs must be a positive integer.")
            return
        if num_samples < 1:
            self.error_stats_label.config(text="Number of configs must be a positive integer.")
            return

        sampling_settings = self.data_gen_settings.camera_config_sampling_settings
        errors = []
        for _ in range(num_samples):
            intrinsics, extrinsics = proj.sample_camera_config(self._rng, self.base_intrinsics, self.base_extrinsics, sampling_settings)
            errors.append(self._keypoint_rms_error(intrinsics.astype(np.float64), extrinsics.astype(np.float64)))
        errors = np.array(errors)

        num_invalid = int(np.sum(np.isnan(errors))) # Configs with a keypoint behind the camera
        valid_errors = errors[~np.isnan(errors)]
        base_error = self._keypoint_rms_error(self.base_intrinsics, self.base_extrinsics)

        if len(valid_errors) == 0:
            self.error_stats_label.config(text=f"All {num_samples} sampled configs had a keypoint behind the camera.")
            return

        self.error_stats_label.config(text=(
            f"Keypoint RMS error over {len(valid_errors)} sampled configs:\n"
            f"  mean {np.mean(valid_errors):.2f} px  (std {np.std(valid_errors):.2f})\n"
            f"  median {np.median(valid_errors):.2f}  min {np.min(valid_errors):.2f}  max {np.max(valid_errors):.2f} px\n"
            f"  base config: {base_error:.2f} px"
            + (f"\n  {num_invalid} config(s) skipped (keypoint behind camera)" if num_invalid else "")
        ))

    def _keypoint_rms_error(self, intrinsics, extrinsics):
        """
        RMS distance (px) between the labelled keypoints and their projections through the given camera; nan if any keypoint is behind the camera
        """
        projected_keypoints, keypoint_depths = _project_with_depth(self.keypoint_world_coords, proj.create_projection_matrix(intrinsics, extrinsics))
        if not np.all(keypoint_depths > MIN_DEPTH):
            return float("nan")
        return _rms_reprojection_error(self.labelled_keypoints, projected_keypoints)

    def _print_config(self):
        intrinsics, extrinsics = self._current_config()
        print(f"Intrinsics: \n{intrinsics}")
        print(f"Extrinsics: \n{extrinsics}")
        print(f"Camera position (in world coords): \n{_camera_position_from_extrinsics(extrinsics)}")

    # MARK: Rendering
    def _schedule_render(self):
        if self._pending_render_after_id is None:
            self._pending_render_after_id = self.after(SLIDER_THROTTLE_MS, self._render)

    def _get_sample_trajectory(self):
        if self._sample_trajectory is None:
            simulation_timestep = settings.SyntheticDataGeneration().trajectory_generation_settings.simulation_timestep
            trajectory = generate_sample_trajectory()[0] # (trajectory, times, eot, bounce_tokens)
            step = max(1, int(round((1.0 / settings.SyntheticDataGeneration().target_fps) / simulation_timestep))) # Only target_fps frames are ever seen by the model
            self._sample_trajectory = trajectory[::step]
        return self._sample_trajectory

    def _render(self):
        self._pending_render_after_id = None

        intrinsics, extrinsics = self._current_config()
        projection_matrix = proj.create_projection_matrix(intrinsics, extrinsics)

        canvas = self._base_image.copy()

        if self.show_base.get():
            for start, end in self.court_segments + self.net_segments:
                _draw_segment(canvas, start, end, self.base_projection_matrix, _BASE_COURT_COLOUR, 2)

        for start, end in self.court_segments:
            _draw_segment(canvas, start, end, projection_matrix, _COURT_COLOUR, 3)
        for start, end in self.net_segments:
            _draw_segment(canvas, start, end, projection_matrix, _NET_COLOUR, 3)

        projected_keypoints, keypoint_depths = _project_with_depth(self.keypoint_world_coords, projection_matrix)
        rms_error = self._keypoint_rms_error(intrinsics, extrinsics)

        if self.show_keypoints.get():
            for labelled, projected, depth in zip(self.labelled_keypoints, projected_keypoints, keypoint_depths):
                labelled_centre = tuple(int(round(v)) for v in labelled)
                cv2.circle(canvas, labelled_centre, 8, _LABELLED_KEYPOINT_COLOUR, -1, lineType=cv2.LINE_AA)
                if depth > MIN_DEPTH:
                    projected_centre = tuple(int(round(v)) for v in np.clip(projected, -1e5, 1e5))
                    cv2.circle(canvas, projected_centre, 8, _PROJECTED_KEYPOINT_COLOUR, 2, lineType=cv2.LINE_AA)
                    cv2.line(canvas, labelled_centre, projected_centre, _PROJECTED_KEYPOINT_COLOUR, 1, cv2.LINE_AA)

        if self.show_trajectory.get():
            trajectory = self._get_sample_trajectory()
            _, depths = _project_with_depth(trajectory, projection_matrix)
            visible_trajectory = trajectory[depths > MIN_DEPTH]
            if len(visible_trajectory) > 1:
                projected_trajectory = proj.project_trajectory_to_2d(visible_trajectory, projection_matrix)
                canvas = overlay_trajectory_on_image(image_path=None, projected_trajectory=np.clip(projected_trajectory, -1e5, 1e5), base_image=canvas)

        display_image = cv2.resize(canvas, DISPLAY_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        self._image_artist.set_data(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        self.canvas.draw_idle()

        self._update_labels(intrinsics, extrinsics, rms_error)

    def _update_labels(self, intrinsics, extrinsics, rms_error):
        for key, (value_label, fmt) in self.value_labels.items():
            value_label.config(text=fmt.format(self.param_vars[key].get()))

        camera_position = _camera_position_from_extrinsics(extrinsics)
        position_diff = camera_position - self.base_camera_position
        intrinsics_diff = intrinsics - self.base_intrinsics
        jitter = self.data_gen_settings.camera_config_sampling_settings

        self.info_label.config(text=(
            f"Camera position: ({camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f}) m\n"
            f"  diff from base: ({position_diff[0]:+.2f}, {position_diff[1]:+.2f}, {position_diff[2]:+.2f}) m\n"
            f"Intrinsics diff from base:\n"
            f"  fx {intrinsics_diff[0, 0]:+.1f}  fy {intrinsics_diff[1, 1]:+.1f}  cx {intrinsics_diff[0, 2]:+.1f}  cy {intrinsics_diff[1, 2]:+.1f} px\n"
            f"Horizontal FOV: {np.rad2deg(2 * np.arctan(self.data_gen_settings.expected_image_size[0] / (2 * intrinsics[0, 0]))):.1f} deg\n"
            f"Keypoint RMS reprojection error: {rms_error:.2f} px\n"
            f"Sampling jitter (settings): +/-{jitter.intrinsics_jitter} px, +/-{jitter.translation_jitter} m, +/-{jitter.rotation_jitter_deg} deg"
        ))


if __name__ == "__main__":
    app = CamConfigExplorerApp()
    app.mainloop()
