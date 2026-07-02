from data_converter import export_annotated_frames_in_yolo_format
import os
from paths import ProjectPaths
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
    output_annotation_path = os.path.join(os.getcwd(), paths.data_folder, numeric_match_folder, clip_folder_name)
    
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
