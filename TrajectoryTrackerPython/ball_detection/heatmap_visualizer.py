from ball_detection.paths import ProjectPaths
import cv2
import matplotlib.pyplot as plt
import os

paths = ProjectPaths()
heatmap = cv2.imread(os.path.join(paths.processed_data_folder, "clip_0001", "heatmaps", "frame_0005_heatmap.png"), cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(8, 5))
plt.imshow(heatmap, cmap='gray')
plt.colorbar(label='Pixel Intensity')

plt.axvline(x=320, color='red', linestyle='--', alpha=0.5, label='Center X (320)')
plt.axhline(y=180, color='red', linestyle='--', alpha=0.5, label='Center Y (180)')

plt.title("Heatmap")
plt.xlabel("Width (Pixels)")
plt.ylabel("Height (Pixels)")
plt.legend()
plt.show()