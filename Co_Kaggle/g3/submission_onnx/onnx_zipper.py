# this script is to going through each subfolders
# zip all onnx files in to one zip file(w1, w2, ...)
# the zip file should land in submission_wzips
# in the end, it should also print out the count of onnx files in the zip file
# and the distribution of onnx files, such as how many been zipped from 05_fill_additive_marking_local_3x3, etc

import os
import zipfile
import re
from collections import Counter

# Determine paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SUBMISSION_ONNX_DIR = SCRIPT_DIR
SUBMISSION_WZIPS_DIR = os.path.join(PROJECT_ROOT, "submission_wzips")

def get_next_zip_name(wzips_dir):
    """
    Finds existing w*.zip files in wzips_dir and returns the next w{n}.zip name.
    """
    existing_numbers = []
    if os.path.exists(wzips_dir):
        for f in os.listdir(wzips_dir):
            match = re.match(r'^w(\d+)\.zip$', f, re.IGNORECASE)
            if match:
                existing_numbers.append(int(match.group(1)))
    
    next_num = max(existing_numbers) + 1 if existing_numbers else 1
    return f"w{next_num}.zip"

def main():
    if not os.path.exists(SUBMISSION_ONNX_DIR):
        print(f"Error: submission_onnx directory not found at {SUBMISSION_ONNX_DIR}")
        return

    # Ensure submission_wzips directory exists
    os.makedirs(SUBMISSION_WZIPS_DIR, exist_ok=True)

    # 1. Collect all ONNX files and their source subfolders
    onnx_files = [] # list of (file_path, folder_name, filename)
    distribution = Counter()

    # List all items in submission_onnx and find subdirectories
    for item in os.listdir(SUBMISSION_ONNX_DIR):
        item_path = os.path.join(SUBMISSION_ONNX_DIR, item)
        if os.path.isdir(item_path):
            # Scan inside this subdirectory for .onnx files
            for f in os.listdir(item_path):
                if f.lower().endswith('.onnx'):
                    f_path = os.path.join(item_path, f)
                    onnx_files.append((f_path, item, f))
                    distribution[item] += 1

    if not onnx_files:
        print(f"No ONNX files found in subfolders of {SUBMISSION_ONNX_DIR}")
        return

    # 2. Get the next wrapper zip filename (w1.zip, w2.zip, ...)
    zip_filename = get_next_zip_name(SUBMISSION_WZIPS_DIR)
    zip_path = os.path.join(SUBMISSION_WZIPS_DIR, zip_filename)

    print(f"Zipping {len(onnx_files)} ONNX file(s) into {zip_filename}...")
    print("-" * 60)

    # 3. Create the zip file and write all files flat
    try:
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for f_path, folder_name, filename in sorted(onnx_files, key=lambda x: (x[1], x[2])):
                zf.write(f_path, arcname=filename)
                print(f"  + Added: {folder_name}/{filename} -> {filename}")
        
        print("-" * 60)
        print(f"Successfully created wrapper zip file: {os.path.join('submission_wzips', zip_filename)}")
        print(f"Total ONNX files zipped: {len(onnx_files)}")
        print("\nDistribution of ONNX files:")
        for folder, count in sorted(distribution.items()):
            print(f"  - {folder}: {count}")
            
    except Exception as e:
        print(f"Error creating zip file {zip_filename}: {e}")

if __name__ == "__main__":
    main()
