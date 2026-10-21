from convert_2d_to_3d.paths import Paths
from convert_2d_to_3d import projector as proj
from convert_2d_to_3d.visualizer import overlay_trajectory_on_image, plot_2d_trajectory, plot_3d_trajectory
import cv2
import numpy as np
import os
import pandas as pd
import time
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

DISPLAY_IMAGE_SIZE = (640, 360) # (width, height); matches the resolution real inference video frames are resized to
RENDER_DEBOUNCE_MS = 80 # delay before rendering a tree selection, so rapid clicking only renders the settled-on item

PLAYBACK_INTERVAL_MS = 15
SLIDER_THROTTLE_MS = 60 # cap redraw rate while dragging the frame slider, for the same reason

FADE_TRAIL_SECONDS = 0.5 # how much recent trajectory history "Fade trail" keeps visible before it's fully faded out


def _dataset_csv_path(paths: Paths):
    return os.path.join(paths.processed_data_folder, "dataset.csv")


def _camera_position_from_extrinsics(extrinsics: np.ndarray):
    r_matrix = extrinsics[0:3, 0:3]
    t_vec = extrinsics[0:3, 3]
    return -r_matrix.T @ t_vec # t = -RT -> T = -R^-1 @ t = -R^T @ t


class DatasetExplorerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Synthetic Dataset Explorer")
        self.geometry("1200x800")

        self.paths = Paths()
        self.df = None
        self._pending_render_after_id = None

        self.video_mode = tk.BooleanVar(value=False)
        self.fade_mode = tk.BooleanVar(value=False)
        self._anim_after_id = None
        self._anim_frame_idx = 0
        self._last_tick_time = None
        self._current_trajectory_row = None
        self._current_trajectory_3d = None
        self._current_trajectory_2d = None
        self._current_trajectory_eot = None
        self._current_trajectory_2d_small = None
        self._fade_window_frames = None # FADE_TRAIL_SECONDS converted to a frame count using the current trajectory's own timestep

        # fig_2d's layout (fixed axis ranges/labels/legend) doesn't change between frames, so tight_layout() only needs to run once rather than on every redraw (see _draw_frame)
        self._layout_initialized = False

        self._last_drawn_frame_idx = -1
        self._pending_slider_after_id = None
        self._pending_slider_idx = 0

        # Base camera config that every row's camera config is a jittered sample of (see projector.sample_camera_config), cached once for the "difference from baseline" panel
        self.base_intrinsics, self.base_extrinsics = proj.get_base_camera_config()
        self.base_camera_position = _camera_position_from_extrinsics(self.base_extrinsics)

        # The sample court image is only ever used at DISPLAY_IMAGE_SIZE resolution here, so it's loaded and downsized once instead of being re-read/re-decoded from disk on every click
        full_image = cv2.imread(self.paths.sample_image_path)
        self._base_image_small = cv2.resize(full_image, DISPLAY_IMAGE_SIZE)

        self._build_widgets()
        self._reload_dataset()

    def _build_widgets(self):
        # Left: hierarchical camera config / trajectory picker
        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        reload_button = ttk.Button(left_frame, text="Reload dataset", command=self._reload_dataset)
        reload_button.pack(fill=tk.X, pady=(0, 4))

        tree_scroll = ttk.Scrollbar(left_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(left_frame, yscrollcommand=tree_scroll.set, selectmode="browse")
        self.tree.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.heading("#0", text="Camera configs / trajectories")
        tree_scroll.config(command=self.tree.yview)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Right: status + plots + metadata
        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        status_frame = ttk.Frame(right_frame)
        status_frame.pack(side=tk.TOP, fill=tk.X)

        self.status_label = ttk.Label(status_frame, text="", foreground="blue")
        self.status_label.pack(side=tk.LEFT, anchor=tk.W)

        video_toggle = ttk.Checkbutton(status_frame, text="Play as video", variable=self.video_mode, command=self._on_video_mode_toggle)
        video_toggle.pack(side=tk.RIGHT)

        fade_toggle = ttk.Checkbutton(status_frame, text="Fade trail", variable=self.fade_mode, command=self._on_fade_mode_toggle)
        fade_toggle.pack(side=tk.RIGHT)

        playback_frame = ttk.Frame(right_frame)
        playback_frame.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        ttk.Label(playback_frame, text="Frame:").pack(side=tk.LEFT)

        self.frame_slider_var = tk.DoubleVar(value=0)
        self.frame_slider = ttk.Scale(playback_frame, from_=0, to=0, orient=tk.HORIZONTAL,
                                       variable=self.frame_slider_var, command=self._on_slider_drag)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        self.frame_slider.bind("<ButtonPress-1>", self._on_slider_press)
        self.frame_slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self.frame_slider.state(["disabled"])

        self.frame_counter_label = ttk.Label(playback_frame, text="- / -", width=12)
        self.frame_counter_label.pack(side=tk.LEFT)

        plots_frame = ttk.Frame(right_frame)
        plots_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig_3d = Figure(figsize=(5, 5))
        self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=plots_frame)
        self.canvas_3d.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Stacked 2D column: the abstract projector.py-style plot on top, the image overlay below it.
        self.fig_2d = Figure(figsize=(5, 8))
        self.ax_2d_plot = self.fig_2d.add_subplot(211)
        self.ax_2d_overlay = self.fig_2d.add_subplot(212)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=plots_frame)
        self.canvas_2d.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.metadata_label = ttk.Label(right_frame, text="Select a trajectory to see its details.", justify=tk.LEFT, font=("TkDefaultFont", 11))
        self.metadata_label.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

        self.diff_label = ttk.Label(right_frame, text="", justify=tk.LEFT, font=("TkDefaultFont", 11))
        self.diff_label.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

    def _reload_dataset(self):
        csv_path = _dataset_csv_path(self.paths)
        if not os.path.exists(csv_path):
            messagebox.showerror("Dataset not found", f"No dataset.csv found at:\n{csv_path}\n\nRun dataset_creator.py first.")
            return

        self.df = pd.read_csv(csv_path)
        self._populate_tree()

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())

        for camera_config_id, config_rows in self.df.groupby("camera_config_id"):
            first_row = config_rows.iloc[0]
            config_label = (f"Camera config {camera_config_id} "
                             f"(fx={first_row.fx:.1f}, fy={first_row.fy:.1f}, "
                             f"cx={first_row.cx:.1f}, cy={first_row.cy:.1f})")
            config_node = self.tree.insert("", tk.END, text=config_label, values=("camera_config", camera_config_id, -1))

            for _, row in config_rows.iterrows():
                trajectory_label = (f"Trajectory {row.trajectory_id} "
                                     f"(num_points={row.num_points}, timestep={row.timestep}, bounces={row.bounces})")
                self.tree.insert(config_node, tk.END, text=trajectory_label, values=("trajectory", camera_config_id, row.trajectory_id))

    def _on_tree_select(self, event):
        selected = self.tree.selection()
        if not selected:
            return


        if self.video_mode.get():
            self.video_mode.set(False)
            self._cancel_animation()
            self._cancel_pending_slider_redraw()

        # Debounce: rapid clicking through several tree items should only render the one the user actually settles on, not every intermediate selection
        if self._pending_render_after_id is not None:
            self.after_cancel(self._pending_render_after_id)
        self._pending_render_after_id = self.after(RENDER_DEBOUNCE_MS, self._render_selection, selected[0])

    def _render_selection(self, item_id):
        self._pending_render_after_id = None

        node_type, camera_config_id, trajectory_id = self.tree.item(item_id, "values")
        camera_config_id = int(camera_config_id)

        self.status_label.config(text="Rendering…")
        self.update_idletasks() # force the status text to actually paint before the render work below

        config_rows = self.df[self.df.camera_config_id == camera_config_id]
        first_row = config_rows.iloc[0]
        extrinsics = first_row[[f"extrinsics_{i}" for i in range(16)]].to_numpy(dtype=float).reshape(4, 4)
        camera_position = _camera_position_from_extrinsics(extrinsics)

        if node_type == "trajectory":
            trajectory_id = int(trajectory_id)
            row = config_rows[config_rows.trajectory_id == trajectory_id].iloc[0]
            self._render_trajectory(row)
            self._update_metadata(row, camera_position, has_trajectory=True)
        else:
            self._cancel_animation()
            self._cancel_pending_slider_redraw()
            self._current_trajectory_row = None
            self._set_slider_enabled(False)
            self.frame_slider.config(to=0)
            self.frame_slider_var.set(0)
            self.frame_counter_label.config(text="- / -")
            self._update_metadata(first_row, camera_position, has_trajectory=False)

        self._update_diff_metadata(first_row, camera_position)
        self.update_idletasks() # force the "Rendering…" text to clear right away rather than on the next idle tick
        self.status_label.config(text="")

    def _render_trajectory(self, row):
        self._cancel_animation()
        self._cancel_pending_slider_redraw()

        data_folder = self.paths.processed_data_folder
        self._current_trajectory_row = row
        self._current_trajectory_3d = np.load(os.path.join(data_folder, row.trajectory_3d_path))
        trajectory_2d = np.load(os.path.join(data_folder, row.trajectory_2d_path))
        self._current_trajectory_2d = trajectory_2d
        self._current_trajectory_eot = np.load(os.path.join(data_folder, row.trajectory_eot_path))

        # Scale projected points down to the cached small overlay image's resolution.
        scale = np.array([self._base_image_small.shape[1] / 1920, self._base_image_small.shape[0] / 1080])
        self._current_trajectory_2d_small = trajectory_2d * scale

        timestep = float(row.timestep)
        self._fade_window_frames = max(1, round(FADE_TRAIL_SECONDS / timestep)) if timestep > 0 else None

        num_points = len(self._current_trajectory_3d)
        self.frame_slider.config(to=max(0, num_points - 1))
        self._set_slider_enabled(True)

        if self.video_mode.get():
            self._start_animation()
        else:
            self._draw_frame(num_points - 1)

    def _draw_frame(self, frame_idx):
        traj_3d = self._current_trajectory_3d[:frame_idx + 1]
        traj_2d = self._current_trajectory_2d[:frame_idx + 1]
        traj_2d_small = self._current_trajectory_2d_small[:frame_idx + 1]
        traj_eot = self._current_trajectory_eot[:frame_idx + 1]
        fade_window = self._fade_window_frames if self.fade_mode.get() else None

        self.ax_3d.clear()
        plot_3d_trajectory(traj_3d, traj_eot, ax=self.ax_3d, fade_window=fade_window)

        self.ax_2d_plot.clear()
        plot_2d_trajectory(traj_2d, ax=self.ax_2d_plot, fade_window=fade_window)

        overlay_img = overlay_trajectory_on_image(self.paths.sample_image_path, traj_2d_small, output_path=None, base_image=self._base_image_small, fade_window=fade_window)
        self.ax_2d_overlay.clear()
        if overlay_img is not None:
            self.ax_2d_overlay.imshow(overlay_img[:, :, ::-1]) # BGR -> RGB
        self.ax_2d_overlay.set_title("Overlaid on sample image")
        self.ax_2d_overlay.axis("off")


        if not self._layout_initialized:
            self.fig_2d.tight_layout()
            self._layout_initialized = True

        self.canvas_3d.draw()
        self.canvas_2d.draw()

        self._last_drawn_frame_idx = frame_idx
        self.frame_slider_var.set(frame_idx)
        num_points = len(self._current_trajectory_3d)
        self.frame_counter_label.config(text=f"{frame_idx} / {num_points - 1}")

    def _on_video_mode_toggle(self):
        self._cancel_animation()
        self._cancel_pending_slider_redraw()
        if self._current_trajectory_row is None:
            return

        if self.video_mode.get():
            self._start_animation()
        else:
            self._draw_frame(len(self._current_trajectory_3d) - 1)

    def _on_fade_mode_toggle(self):
        if self._current_trajectory_row is not None:
            self._draw_frame(self._last_drawn_frame_idx)

    def _set_slider_enabled(self, enabled: bool):
        self.frame_slider.state(["!disabled"] if enabled else ["disabled"])

    def _on_slider_press(self, event):
        self._cancel_animation()

    def _on_slider_drag(self, value_str):
        if self._current_trajectory_row is None:
            return

        new_idx = int(round(float(value_str)))
        if new_idx == self._last_drawn_frame_idx:
            return # this call is _draw_frame syncing the slider's own position, not a user drag

        self._pending_slider_idx = new_idx
        if self._pending_slider_after_id is None:
            self._pending_slider_after_id = self.after(SLIDER_THROTTLE_MS, self._flush_slider_drag)

    def _flush_slider_drag(self):
        self._pending_slider_after_id = None
        self._anim_frame_idx = self._pending_slider_idx
        self._draw_frame(self._pending_slider_idx)

    def _cancel_pending_slider_redraw(self):
        if self._pending_slider_after_id is not None:
            self.after_cancel(self._pending_slider_after_id)
            self._pending_slider_after_id = None

    def _on_slider_release(self, event):
        if self._pending_slider_after_id is not None:
            self._cancel_pending_slider_redraw()
            self._anim_frame_idx = self._pending_slider_idx
            self._draw_frame(self._pending_slider_idx)

        # Resume playback from wherever the user scrubbed to, rather than restarting from frame 0.
        if self.video_mode.get() and self._current_trajectory_row is not None:
            self._start_animation(reset_to_start=False)

    def _cancel_animation(self):
        if self._anim_after_id is not None:
            self.after_cancel(self._anim_after_id)
            self._anim_after_id = None

    def _start_animation(self, reset_to_start=True):
        self._cancel_animation()
        if self._current_trajectory_row is None:
            return

        if reset_to_start:
            self._anim_frame_idx = 0
        self._last_tick_time = time.perf_counter()
        self._animation_step()

    def _animation_step(self):
        self._draw_frame(self._anim_frame_idx)

        now = time.perf_counter()
        elapsed_ms = (now - self._last_tick_time) * 1000
        self._last_tick_time = now

        num_points = len(self._current_trajectory_3d)
        timestep_ms = float(self._current_trajectory_row.timestep) * 1000
        frames_to_advance = max(1, round(elapsed_ms / timestep_ms))
        self._anim_frame_idx += frames_to_advance
        if self._anim_frame_idx >= num_points:
            self._anim_frame_idx = 0 # loop back to the start

        self._anim_after_id = self.after(PLAYBACK_INTERVAL_MS, self._animation_step)

    def _update_metadata(self, row, camera_position, has_trajectory: bool):
        lines = [
            f"Camera config {row.camera_config_id}:",
            f"  fx={row.fx:.2f}  fy={row.fy:.2f}  cx={row.cx:.2f}  cy={row.cy:.2f}",
            f"  Camera position (world coords): "
            f"({camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f})",
        ]
        if has_trajectory:
            lines.append(f"Trajectory {row.trajectory_id}:")
            lines.append(f"  num_points={row.num_points}  timestep={row.timestep}  bounces={row.bounces}")

        self.metadata_label.config(text="\n".join(lines))

    def _update_diff_metadata(self, row, camera_position):
        d_fx = row.fx - self.base_intrinsics[0, 0]
        d_fy = row.fy - self.base_intrinsics[1, 1]
        d_cx = row.cx - self.base_intrinsics[0, 2]
        d_cy = row.cy - self.base_intrinsics[1, 2]
        d_position = camera_position - self.base_camera_position

        lines = [
            "Difference vs. baseline camera config:",
            f"  Δfx={d_fx:+.2f}  Δfy={d_fy:+.2f}  Δcx={d_cx:+.2f}  Δcy={d_cy:+.2f}",
            f"  Δposition (world coords): "
            f"({d_position[0]:+.3f}, {d_position[1]:+.3f}, {d_position[2]:+.3f})",
        ]
        self.diff_label.config(text="\n".join(lines))


if __name__ == "__main__":
    app = DatasetExplorerApp()
    app.mainloop()
