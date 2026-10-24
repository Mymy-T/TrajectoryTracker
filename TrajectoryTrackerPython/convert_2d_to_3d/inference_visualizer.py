from convert_2d_to_3d.paths import Paths
from convert_2d_to_3d.settings import SyntheticDataGeneration
from convert_2d_to_3d.visualizer import plot_3d_trajectory
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import os
import time
import tkinter as tk
from tkinter import ttk

# python3 -m convert_2d_to_3d.inference_visualizer (run after running inference to save trajectory)

PLAYBACK_INTERVAL_MS = 15
SLIDER_THROTTLE_MS = 60 # cap redraw rate while dragging the step slider
THREE_D_ZOOM = 1.0

PRED_COLOR = "orange"
ACTUAL_COLOR = "blue"
FAINT_PRED_COLOR = "navajowhite"
FAINT_ACTUAL_COLOR = "lightsteelblue"


def load_trajectory(trajectory_path):
    """
    Returns (coords, eot, coord_labels, eot_labels), or None if the trajectory hasn't been saved yet
    Written by inference.py's save_trajectory_3d(); coords/coord_labels are (L, 3), eot/eot_labels are (L,)
    """
    if not os.path.exists(trajectory_path):
        return None

    data = np.load(trajectory_path)
    return data["coords"], data["eot"], data["coord_labels"], data["eot_labels"]


class InferenceVisualizerApp(tk.Tk):
    def __init__(self, pred_coords, pred_eot, coord_labels, eot_labels, fps):
        super().__init__()
        self.title("Inference Visualizer (Synthetic Test)")
        self.geometry("1500x650")

        self.pred_coords = pred_coords
        self.pred_eot = pred_eot
        self.coord_labels = coord_labels
        self.eot_labels = eot_labels
        self.num_steps = len(pred_coords)
        self.fps = fps

        self._anim_after_id = None
        self._anim_step_idx = 0
        self._last_tick_time = None
        self._last_drawn_step_idx = -1
        self._pending_slider_after_id = None
        self._pending_slider_idx = 0

        self._build_widgets()
        self._draw_step(0)

    def _build_widgets(self):
        self.fig = Figure(figsize=(15, 5))
        self.ax_pred = self.fig.add_subplot(1, 3, 1, projection="3d")
        self.ax_actual = self.fig.add_subplot(1, 3, 2, projection="3d")
        self.ax_overlay = self.fig.add_subplot(1, 3, 3, projection="3d")

        # Full paths drawn faintly once for context; the trails/markers below are what move
        plot_3d_trajectory(self.pred_coords, self.pred_eot, ax=self.ax_pred, color=FAINT_PRED_COLOR, label="Prediction")
        plot_3d_trajectory(self.coord_labels, self.eot_labels, ax=self.ax_actual, color=FAINT_ACTUAL_COLOR, label="Actual")
        plot_3d_trajectory(self.coord_labels, self.eot_labels, ax=self.ax_overlay, color=FAINT_ACTUAL_COLOR, label="Overlay", hide_legend=True)
        plot_3d_trajectory(self.pred_coords, self.pred_eot, ax=self.ax_overlay, color=FAINT_PRED_COLOR, label="Overlay", hide_legend=True)

        for ax in (self.ax_pred, self.ax_actual, self.ax_overlay):
            ax.set_box_aspect(ax.get_box_aspect(), zoom=THREE_D_ZOOM) # plot_3d_trajectory's court-proportioned box leaves most of each panel empty

        # NOTE: plot order is (x, z, y), matching plot_3d_trajectory, since +z is up in matplotlib but +y is up in trajectory coords
        self.pred_trail, self.pred_marker = self._create_trail_and_marker(self.ax_pred, PRED_COLOR)
        self.actual_trail, self.actual_marker = self._create_trail_and_marker(self.ax_actual, ACTUAL_COLOR)
        self.overlay_actual_trail, self.overlay_actual_marker = self._create_trail_and_marker(self.ax_overlay, ACTUAL_COLOR)
        self.overlay_pred_trail, self.overlay_pred_marker = self._create_trail_and_marker(self.ax_overlay, PRED_COLOR)
        self.error_segment, = self.ax_overlay.plot([], [], [], color="gray", linewidth=1) # Joins the current predicted and actual points

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        playback_frame = ttk.Frame(self)
        playback_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        self.play_button = ttk.Button(playback_frame, text="Play", command=self._on_play_toggle)
        self.play_button.pack(side=tk.LEFT)

        self.step_slider_var = tk.DoubleVar(value=0)
        self.step_slider = ttk.Scale(playback_frame, from_=0, to=self.num_steps - 1, orient=tk.HORIZONTAL,
                                      variable=self.step_slider_var, command=self._on_slider_drag, takefocus=False) # takefocus off so arrow keys only step once (see _step)
        self.step_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self.step_slider.bind("<ButtonPress-1>", self._on_slider_press)
        self.step_slider.bind("<ButtonRelease-1>", self._on_slider_release)

        self.step_counter_label = ttk.Label(playback_frame, text="", width=20)
        self.step_counter_label.pack(side=tk.LEFT)

        # Step one timestep at a time
        self.bind("<Left>", lambda event: self._step(-1))
        self.bind("<Right>", lambda event: self._step(1))

    def _create_trail_and_marker(self, ax, color):
        trail, = ax.plot([], [], [], color=color, linewidth=2)
        marker, = ax.plot([], [], [], "o", color=color, markeredgecolor="black", markersize=8)
        return trail, marker

    def _update_trail_and_marker(self, trail, marker, coords, step_idx):
        visible = coords[:step_idx + 1]
        trail.set_data_3d(visible[:, 0], visible[:, 2], visible[:, 1])
        marker.set_data_3d([coords[step_idx, 0]], [coords[step_idx, 2]], [coords[step_idx, 1]])

    def _draw_step(self, step_idx):
        self._update_trail_and_marker(self.pred_trail, self.pred_marker, self.pred_coords, step_idx)
        self._update_trail_and_marker(self.actual_trail, self.actual_marker, self.coord_labels, step_idx)
        self._update_trail_and_marker(self.overlay_pred_trail, self.overlay_pred_marker, self.pred_coords, step_idx)
        self._update_trail_and_marker(self.overlay_actual_trail, self.overlay_actual_marker, self.coord_labels, step_idx)

        pred_point = self.pred_coords[step_idx]
        actual_point = self.coord_labels[step_idx]
        self.error_segment.set_data_3d([pred_point[0], actual_point[0]], [pred_point[2], actual_point[2]], [pred_point[1], actual_point[1]])

        error = np.linalg.norm(pred_point - actual_point)
        self.ax_pred.set_title(f"Prediction (height {pred_point[1]:.2f} m)")
        self.ax_actual.set_title(f"Actual (height {actual_point[1]:.2f} m)")
        self.ax_overlay.set_title(f"Overlay (error {error:.2f} m)")

        self.canvas.draw_idle()

        self._last_drawn_step_idx = step_idx
        self.step_slider_var.set(step_idx)
        self.step_counter_label.config(text=f"{step_idx} / {self.num_steps - 1}  ({step_idx / self.fps:.2f} s)")

    def _step(self, step):
        self._stop_animation()
        self._cancel_pending_slider_redraw()
        new_idx = min(max(self._last_drawn_step_idx + step, 0), self.num_steps - 1)
        self._anim_step_idx = new_idx
        self._draw_step(new_idx)

    def _on_play_toggle(self):
        if self._anim_after_id is not None:
            self._stop_animation()
        else:
            self._start_animation()

    def _on_slider_press(self, event):
        self._stop_animation()

    def _on_slider_drag(self, value_str):
        new_idx = int(round(float(value_str)))
        if new_idx == self._last_drawn_step_idx:
            return # this call is _draw_step syncing the slider's own position, not a user drag

        self._pending_slider_idx = new_idx
        if self._pending_slider_after_id is None:
            self._pending_slider_after_id = self.after(SLIDER_THROTTLE_MS, self._flush_slider_drag)

    def _flush_slider_drag(self):
        self._pending_slider_after_id = None
        self._anim_step_idx = self._pending_slider_idx
        self._draw_step(self._pending_slider_idx)

    def _cancel_pending_slider_redraw(self):
        if self._pending_slider_after_id is not None:
            self.after_cancel(self._pending_slider_after_id)
            self._pending_slider_after_id = None

    def _on_slider_release(self, event):
        if self._pending_slider_after_id is not None:
            self._cancel_pending_slider_redraw()
            self._anim_step_idx = self._pending_slider_idx
            self._draw_step(self._pending_slider_idx)

    def _stop_animation(self):
        if self._anim_after_id is not None:
            self.after_cancel(self._anim_after_id)
            self._anim_after_id = None
        self.play_button.config(text="Play")

    def _start_animation(self):
        self._stop_animation()
        self.play_button.config(text="Pause")
        self._last_tick_time = time.perf_counter()
        self._animation_step()

    def _animation_step(self):
        self._draw_step(self._anim_step_idx)

        # Advance by however many timesteps of real time have passed, so playback stays at the trajectory's fps even when drawing is slow
        now = time.perf_counter()
        elapsed_ms = (now - self._last_tick_time) * 1000
        self._last_tick_time = now

        step_duration_ms = 1000 / self.fps
        steps_to_advance = max(1, round(elapsed_ms / step_duration_ms))
        self._anim_step_idx += steps_to_advance
        if self._anim_step_idx >= self.num_steps:
            self._anim_step_idx = 0 # loop back to the start

        self._anim_after_id = self.after(PLAYBACK_INTERVAL_MS, self._animation_step)


def main():
    trajectory = load_trajectory(Paths().inference_trajectory_path)
    if trajectory is None:
        raise FileNotFoundError("Couldn't find saved trajectories; run python3 -m convert_2d_to_3d.inference --name <model name> first.")

    fps = SyntheticDataGeneration().target_fps # Trajectories are resampled to this fps

    app = InferenceVisualizerApp(*trajectory, fps)
    app.mainloop()


if __name__ == "__main__":
    main()
