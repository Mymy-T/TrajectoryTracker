import json
import os
from paths import ProjectPaths
import shutil
import subprocess
import sys

def load_or_create_mapping(mapping_file, match_title):
    """
    Loads mapping and ensures a permanent match_#### assignment for this match title.
    """
    mapping = {} # keys are video titles, values are folder_names (match_####)
    max_index = -1

    if os.path.exists(mapping_file):
        with open(mapping_file, "r") as file:
            for line in file:
                if ":" in line:
                    folder_name, title = line.strip().split(":", 1) # since title is on the right, the first ':' will always split the corresponding folder name (of the form match_####) from the title
                    mapping[title] = folder_name
                    try:
                        idx = int(folder_name.split("_")[1])
                        max_index = max(idx, max_index)
                    except (ValueError, IndexError):
                        pass

    if match_title not in mapping:
        next_index = max_index + 1
        mapping[match_title] = f"match_{next_index:04d}"
        with open(mapping_file, "a") as f: # also auto creates mapping.txt if it doesn't exist
            f.write(f"{mapping[match_title]}:{match_title}\n")

    return mapping[match_title] # match_####

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
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    
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



if __name__ == '__main__':

    paths = ProjectPaths()
    
    #  Enforce argument validation (expecting the path directly to a specific clip folder)
    if len(sys.argv) != 2:
        print("Error: Missing target clip directory argument.")
        print("Usage: python3 annotation.py Match_Title/clip_####")
        sys.exit(1)

    source_clip_partial_path = sys.argv[1]
    source_clip_path = os.path.join(os.getcwd(), paths.raw_frame_folder, source_clip_partial_path)

    if not os.path.exists(source_clip_path) or not os.path.isdir(source_clip_path):
        print(f"Error: Clip directory does not exist: {source_clip_path}")
        sys.exit(1)


    match_title = source_clip_partial_path.split("/")[0]
    clip_folder_name = source_clip_partial_path.split("/")[1]

    # Get the numeric match folder (match_####)
    mapping_file = os.path.join(os.getcwd(), "match_title_to_number_mapping.txt")
    numeric_match_folder = load_or_create_mapping(mapping_file, match_title)

    # Generate the corresponding target output directory path
    output_annotation_path = os.path.join(os.getcwd(), paths.annotated_data_folder, numeric_match_folder, clip_folder_name)
    
    # Check if directory already exists
    if os.path.exists(output_annotation_path):
        print(f"Output directory already exists: {numeric_match_folder}/{clip_folder_name}")
    else:
        os.makedirs(output_annotation_path, exist_ok=True)
        print(f"Output directory created: {numeric_match_folder}/{clip_folder_name}")


    command = [
        sys.executable, "-m", "anylabeling.app",
        source_clip_path,
        # "--output", output_annotation_path
    ]

    try:
        subprocess.run(command, check=True)

        print("Anylabeling window closed; exporting annotated frames...")

        export_annotated_frames_in_yolo_format(
            input_dir=source_clip_path,
            output_dir=output_annotation_path
        )
        
        sys.exit(0) 
    except FileNotFoundError:
        print("Error: 'anylabeling' executable could not be verified in terminal environments.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"App termination error reported: {e}")
        sys.exit(1)
