import os
import json
import shutil
import sys

def export_annotated_frames_in_yolo_format(input_dir, output_dir):
    """
    Converts AnyLabeling JSON files to YOLO txt format, preserves the JSON files,
    and organizes the final output into split 'images' and 'labels' directories.
    
    Args:
        input_dir (str): Folder used to open anylabeling
        output_dir (str): The target match_####/clip_#### root destination folder.
    """
    # Define and enforce the standard YOLO structured subdirectories
    images_output_dir = os.path.join(output_dir, "images")
    labels_output_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    class_mapping = {"negative": 0, "ball": 1}
    
    # Gather all JSON annotation files from the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    print(f"Starting YOLO conversion for {len(json_files)} frames...")
    
    for file in json_files:
        json_path = os.path.join(input_dir, file)
        base_name, _ = os.path.splitext(file)
        
        # Define output destinations
        source_frame_path = os.path.join(input_dir, f"{base_name}.jpg")
        label_out_path = os.path.join(labels_output_dir, f"{base_name}.txt")
        class_txt_out_path = os.path.join(output_dir, "classes.txt") # Mimic anylabeling manual export
        
        # Parse JSON coordinate tracking data
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            yolo_lines = []
            img_width = data.get("imageWidth")
            img_height = data.get("imageHeight")
            
            for shape in data.get("shapes", []):
                label = shape.get("label")
                points = shape.get("points", [])
                
                class_id = class_mapping.get(label, -1)
                try:
                    if class_id == -1:
                        raise KeyError
                except KeyError:
                    print(f"KeyError - Invalid class label '{label}' for file: {base_name}")
                    sys.exit(1)
                
                # Verify Bounding Box formatting (2 diagonal coordinate pairs)
                if len(points) == 2:
                    (x1, y1), (x2, y2) = points[0], points[1]
                    
                    # Compute normalized relative width, height, and center spatial positions
                    box_w = abs(x2 - x1)
                    box_h = abs(y2 - y1)
                    x_center = min(x1, x2) + (box_w / 2.0)
                    y_center = min(y1, y2) + (box_h / 2.0)
                    
                    norm_x = x_center / img_width
                    norm_y = y_center / img_height
                    norm_w = box_w / img_width
                    norm_h = box_h / img_height
                    
                    yolo_lines.append(f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")
            
            # Write out the final normalized YOLO text file
            with open(label_out_path, "w", encoding="utf-8") as f:
                f.write(" ".join(yolo_lines))
                
            # Securely copy corresponding image file over to its dedicated folder
            if os.path.exists(source_frame_path):
                shutil.copy2(source_frame_path, images_output_dir)
            else:
                print(f"Source image frame not found: {base_name}.jpg")
                sys.exit(1)
            
            with open(class_txt_out_path, "w", encoding="utf-8") as f:
                lines = []
                for k, v, in class_mapping.items():
                    line = " ".join([k, str(v)]) # k is class label, v is corresponding integer id
                    lines.append(line)
                f.write("\n".join(lines))
                
                
        except Exception as e:
            print(f"Exception: {e}")
            sys.exit(1)
            

    print(f"Success! See {output_dir}")
