import argparse
from convert_2d_to_3d.visualizer import plot_3d_trajectory
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import os
import time
import tkinter as tk
from tkinter import ttk

DISPLAY_IMAGE_SIZE = (640, 360) # (width, height); frames are downsized once on load, so redraws don't touch full 1080p images
PLAYBACK_INTERVAL_MS = 15
SLIDER_THROTTLE_MS = 60 # cap redraw rate while dragging the frame slider
THREE_D_ZOOM = 1.3

TRAJECTORY_FILENAME = "preds.npz" # Written by full_inference.py's save_trajectory_3d(); keys: coords (L, 3), eot (L,), frame_indices (L,)
TRACKED_VIDEO_FILENAME = "annotated_video.mp4"


def load_video_frames(video_path):
    """
    Returns (frames, fps); frames is a list of RGB arrays at DISPLAY_IMAGE_SIZE
    All frames are kept in memory so the slider can jump to any frame instantly (seeking mp4v videos with CAP_PROP_POS_FRAMES is slow and not always frame-exact)
    """
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []

    success, img = vidcap.read()
    while success:
        img = cv2.resize(img, DISPLAY_IMAGE_SIZE)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        success, img = vidcap.read()

    vidcap.release()
    return frames, fps


def load_trajectory(trajectory_path):
    """
    Returns (coords, eot, frame_indices), or None if the trajectory hasn't been saved yet
    frame_indices[i] is the original video frame that coords[i] corresponds to
    """
    if not os.path.exists(trajectory_path):
        return None

    data = np.load(trajectory_path)
    return data["coords"], data["eot"], data["frame_indices"].astype(int)


class InferenceVisualizerApp(tk.Tk):
    def __init__(self, original_frames, tracked_frames, fps, trajectory):
        super().__init__()
        self.title("Inference Visualizer")
        self.geometry("1200x900")

        if len(original_frames) != len(tracked_frames):
            print(f"Warning: original clip has {len(original_frames)} frames but tracked video has {len(tracked_frames)}; only the first {min(len(original_frames), len(tracked_frames))} are shown")

        self.original_frames = original_frames
        self.tracked_frames = tracked_frames
        self.num_frames = min(len(original_frames), len(tracked_frames))
        self.fps = fps
        self.trajectory = trajectory

        # Maps video frame idx -> trajectory row; frames without a 3D estimate (eg the first 2 TrackNet frames) are absent
        self.frame_to_row = {} if trajectory is None else {frame_idx: row for row, frame_idx in enumerate(trajectory[2])}

        self._anim_after_id = None
        self._anim_frame_idx = 0
        self._last_tick_time = None
        self._last_drawn_frame_idx = -1
        self._pending_slider_after_id = None
        self._pending_slider_idx = 0

        self._build_widgets()
        self._draw_frame(0)

    def _build_widgets(self):
        self.fig = Figure(figsize=(12, 8))
        grid = self.fig.add_gridspec(2, 2, height_ratios=[1, 1.3])

        self.ax_original = self.fig.add_subplot(grid[0, 0])
        self.ax_tracked = self.fig.add_subplot(grid[0, 1])
        self.ax_3d = self.fig.add_subplot(grid[1, :], projection="3d")

        # Images are created once and only have their data swapped per frame (see _draw_frame)
        self.original_image = self.ax_original.imshow(self.original_frames[0])
        self.ax_original.set_title("Original clip")
        self.ax_original.axis("off")

        self.tracked_image = self.ax_tracked.imshow(self.tracked_frames[0])
        self.ax_tracked.set_title("Tracked ball")
        self.ax_tracked.axis("off")

        if self.trajectory is None:
            self.ax_3d.text2D(0.5, 0.5, f"{TRAJECTORY_FILENAME} not found; run full_inference.py", transform=self.ax_3d.transAxes, ha="center")
            self.ax_3d.set_axis_off()
        else:
            coords, eot, _ = self.trajectory
            plot_3d_trajectory(coords, eot, ax=self.ax_3d, color="lightgray") # Full path drawn faintly once for context; the trail/marker below are what move
            self.ax_3d.set_box_aspect(self.ax_3d.get_box_aspect(), zoom=THREE_D_ZOOM) # plot_3d_trajectory's court-proportioned box leaves most of the wide panel empty

        # NOTE: plot order is (x, z, y), matching plot_3d_trajectory, since +z is up in matplotlib but +y is up in trajectory coords
        self.trail_line, = self.ax_3d.plot([], [], [], color="orange", linewidth=2)
        self.position_marker, = self.ax_3d.plot([], [], [], "o", color="orange", markeredgecolor="black", markersize=8)

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        playback_frame = ttk.Frame(self)
        playback_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        self.play_button = ttk.Button(playback_frame, text="Play", command=self._on_play_toggle)
        self.play_button.pack(side=tk.LEFT)

        self.frame_slider_var = tk.DoubleVar(value=0)
        self.frame_slider = ttk.Scale(playback_frame, from_=0, to=self.num_frames - 1, orient=tk.HORIZONTAL,
                                       variable=self.frame_slider_var, command=self._on_slider_drag, takefocus=False) # takefocus off so arrow keys only step once (see _step_frame)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self.frame_slider.bind("<ButtonPress-1>", self._on_slider_press)
        self.frame_slider.bind("<ButtonRelease-1>", self._on_slider_release)

        self.frame_counter_label = ttk.Label(playback_frame, text="", width=20)
        self.frame_counter_label.pack(side=tk.LEFT)

        # Step one frame at a time; useful for grabbing exact frames
        self.bind("<Left>", lambda event: self._step_frame(-1))
        self.bind("<Right>", lambda event: self._step_frame(1))

    def _draw_frame(self, frame_idx):
        self.original_image.set_data(self.original_frames[frame_idx])
        self.tracked_image.set_data(self.tracked_frames[frame_idx])

        row = self.frame_to_row.get(frame_idx)
        if row is not None:
            coords = self.trajectory[0]
            trail = coords[:row + 1]
            self.trail_line.set_data_3d(trail[:, 0], trail[:, 2], trail[:, 1])
            self.position_marker.set_data_3d([coords[row, 0]], [coords[row, 2]], [coords[row, 1]])
            self.ax_3d.set_title(f"3D Trajectory (height {coords[row, 1]:.2f} m)")
        elif self.trajectory is not None:
            self.trail_line.set_data_3d([], [], [])
            self.position_marker.set_data_3d([], [], [])
            self.ax_3d.set_title("3D Trajectory (no 3D estimate for this frame)")

        self.canvas.draw_idle()

        self._last_drawn_frame_idx = frame_idx
        self.frame_slider_var.set(frame_idx)
        self.frame_counter_label.config(text=f"{frame_idx} / {self.num_frames - 1}  ({frame_idx / self.fps:.2f} s)")

    def _step_frame(self, step):
        self._stop_animation()
        self._cancel_pending_slider_redraw()
        new_idx = min(max(self._last_drawn_frame_idx + step, 0), self.num_frames - 1)
        self._anim_frame_idx = new_idx
        self._draw_frame(new_idx)

    def _on_play_toggle(self):
        if self._anim_after_id is not None:
            self._stop_animation()
        else:
            self._start_animation()

    def _on_slider_press(self, event):
        self._stop_animation()

    def _on_slider_drag(self, value_str):
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
        self._draw_frame(self._anim_frame_idx)

        # Advance by however many frames of real time have passed, so playback stays at the clip's fps even when drawing is slow
        now = time.perf_counter()
        elapsed_ms = (now - self._last_tick_time) * 1000
        self._last_tick_time = now

        frame_duration_ms = 1000 / self.fps
        frames_to_advance = max(1, round(elapsed_ms / frame_duration_ms))
        self._anim_frame_idx += frames_to_advance
        if self._anim_frame_idx >= self.num_frames:
            self._anim_frame_idx = 0 # loop back to the start

        self._anim_after_id = self.after(PLAYBACK_INTERVAL_MS, self._animation_step)


def main():
    parser = argparse.ArgumentParser(description="View the original clip, the tracked ball video, and the reconstructed 3D trajectory in sync")
    parser.add_argument("--video", default="video", help="Name of the analyzed video within full_inference_results. Don't add extension.")
    args = parser.parse_args()

    inference_base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_inference_results")

    video_path = os.path.join(inference_base_folder, f"{args.video}.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError("Couldn't find video path within full_inference_results.")

    tracked_video_path = os.path.join(inference_base_folder, TRACKED_VIDEO_FILENAME)
    if not os.path.exists(tracked_video_path):
        raise FileNotFoundError("Couldn't find tracked (annotated) video within full_inference_results; run full_inference.py first.")

    original_frames, fps = load_video_frames(video_path)
    tracked_frames, _ = load_video_frames(tracked_video_path)
    trajectory = load_trajectory(os.path.join(inference_base_folder, TRAJECTORY_FILENAME))

    app = InferenceVisualizerApp(original_frames, tracked_frames, fps, trajectory)
    app.mainloop()


if __name__ == "__main__":
    main()
