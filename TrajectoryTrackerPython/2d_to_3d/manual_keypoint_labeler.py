import cv2

selected_points = []

def click_event(event, x, y, flags, params):
    """Callback function that handles mouse click events."""
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        print(f"Point selected: X={x}, Y={y}")
        
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1) # Show labeled point in image
        cv2.imshow("Image Viewer", img)

image_path = "./sample_img.png"
img = cv2.imread(image_path)

if img is None:
    print("Error: Could not open or find the image.")
    exit()

# 2. Create a window and set the mouse callback handler
cv2.namedWindow("Image Viewer")
cv2.setMouseCallback("Image Viewer", click_event)

print("Click on the image to select points. Press 'q' or 'esc' to exit and view final list.")
while True:
    cv2.imshow("Image Viewer", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:  # 27 is esc
        break

cv2.destroyAllWindows()

print("\n--- Final Selected Coordinates ---")
print(selected_points)
