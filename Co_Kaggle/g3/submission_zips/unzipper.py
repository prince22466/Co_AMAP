# this script is to unzip each zip file in folder submission_zips
# put onnx file into corresponding sub folders in submission_onnx
# as shown below

import os
import zipfile
import re

# =====================================================================
# MAPPING SPECIFICATION (Editable Block)
# You can update this list/format from time to time.
# The script will parse it dynamically at runtime.
# =====================================================================
MAPPING_SPEC = r"""
"submission_onnx\05_fill_additive_marking_local_3x3"
[`task015`
`task081`
`task095`
`task220`
`task230`
`task258`
`task331`
`task352`]


"submission_onnx\05_fill_additive_marking_nonlocal_1color"
[`task002`
`task027`
`task042`
`task043`
`task047`
`task050`
`task060`
`task063`
`task090`
`task102`
`task105`
`task119`
`task126`
`task139`
`task162`
`task166`
`task176`
`task200`
`task219`
`task232`
`task246`
`task251`
`task255`
`task265`
`task273`
`task278`
`task299`
`task303`
`task323`
`task335`
`task336`
`task341`
`task348`
`task350`
`task357`
`task367`
`task371`
`task381`
`task387`
`task392`
`task397`]

"submission_onnx\05_fill_additive_marking_nonlocal_multicolor"
[`task055`
`task145`
`task187`
`task198`
`task204`
`task226`
`task256`
`task302`
`task349`
`task369`]


"submission_onnx\09_pattern-continuation-localish_additive"
[`task017`
`task020`
`task041`
`task051`
`task061`
`task110`
`task112`
`task133`
`task168`
`task173`
`task175`
`task243`
`task285`
`task305`
`task378`]


"submission_onnx\09_pattern-continuation-localish_recolor"
[`task025`
`task064`
`task074`
`task093`
`task118`
`task143`
`task158`
`task182`
`task208`
`task228`
`task287`
`task340`]


"submission_onnx\09_pattern-continuation-nonlocal_additive"
[`task005`
`task007`
`task009`
`task012`
`task013`
`task024`
`task028`
`task033`
`task037`
`task045`
`task066`
`task076`
`task080`
`task082`
`task084`
`task089`
`task092`
`task099`
`task101`
`task113`
`task117`
`task132`
`task136`
`task137`
`task141`
`task165`
`task181`
`task190`
`task191`
`task197`
`task212`
`task214`
`task215`
`task217`
`task224`
`task225`
`task237`
`task240`
`task248`
`task268`
`task280`
`task284`
`task286`
`task288`
`task297`
`task306`
`task322`
`task328`
`task333`
`task343`
`task345`
`task356`
`task358`
`task361`
`task363`
`task382`
`task385`]


"submission_onnx\09_pattern-continuation-nonlocal_recolor"
[`task044`
`task054`
`task059`
`task085`
`task094`
`task202`
`task206`
`task279`
`task281`
`task314`
`task324`
`task370`
`task375`
`task379`
`task383`]
"""

# Determine paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SUBMISSION_ZIPS_DIR = os.path.join(PROJECT_ROOT, "submission_zips")
SUBMISSION_ONNX_DIR = os.path.join(PROJECT_ROOT, "submission_onnx")

def parse_mapping(spec_text):
    """
    Parses the mapping format dynamically from the text block.
    """
    folder_tasks = {}
    lines = [line.strip() for line in spec_text.split('\n') if line.strip()]
    
    current_folder = None
    in_bracket = False
    
    for line in lines:
        # Check if line matches a folder name inside quotes (e.g. "submission_onnx\some_folder")
        folder_match = re.match(r'^"?submission_onnx[\\/]([^"]+)"?$', line)
        if folder_match:
            current_folder = folder_match.group(1).strip()
            folder_tasks[current_folder] = []
            continue
        
        if '[' in line:
            in_bracket = True
        
        if in_bracket:
            # Extract task names
            tasks = re.findall(r'task\d+', line, re.IGNORECASE)
            if current_folder and tasks:
                folder_tasks[current_folder].extend([t.lower() for t in tasks])
                
        if ']' in line:
            in_bracket = False
            current_folder = None
            
    return folder_tasks

def main():
    # Verify directories
    if not os.path.exists(SUBMISSION_ZIPS_DIR):
        print(f"Error: submission_zips folder not found at {SUBMISSION_ZIPS_DIR}")
        return
    
    # Verify target directory exists
    if not os.path.exists(SUBMISSION_ONNX_DIR):
        print(f"Error: submission_onnx folder not found at {SUBMISSION_ONNX_DIR}")
        return

    # 1. Dynamically parse target mappings from the raw spec string
    folder_tasks_dict = parse_mapping(MAPPING_SPEC)
    
    # 2. Build reverse lookup map
    task_to_folder = {}
    for folder, tasks in folder_tasks_dict.items():
        for task in tasks:
            task_to_folder[task.lower()] = folder

    # 3. Get list of zip files
    zip_files = [f for f in os.listdir(SUBMISSION_ZIPS_DIR) if f.lower().endswith('.zip')]
    
    if not zip_files:
        print(f"No zip files found in {SUBMISSION_ZIPS_DIR}")
        return

    print(f"Found {len(zip_files)} zip file(s) in {SUBMISSION_ZIPS_DIR}")
    print("-" * 60)

    # Keep track of files that are overwritten
    overwritten_files = []

    # Dictionary to keep track of copied files for duplicate handling
    copied_files = {} # task_id -> (zip_name, file_size, dest_path)

    # 4. Process each zip file
    for zip_name in zip_files:
        zip_path = os.path.join(SUBMISSION_ZIPS_DIR, zip_name)
        print(f"Processing: {zip_name} ...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    
                    filename = os.path.basename(info.filename)
                    if not filename.lower().endswith('.onnx'):
                        continue
                    
                    # Extract the task ID (e.g. task015 from task015.onnx)
                    match = re.search(r'(task\d+)', filename, re.IGNORECASE)
                    if not match:
                        print(f"  [Warning] Could not parse task ID from file name: {filename}")
                        continue
                    
                    task_id = match.group(1).lower()
                    
                    # Look up destination folder
                    folder_name = task_to_folder.get(task_id)
                    if not folder_name:
                        print(f"  [Warning] Task '{task_id}' has no defined destination folder in mapping dictionary.")
                        continue
                    
                    dest_dir = os.path.join(SUBMISSION_ONNX_DIR, folder_name)
                    # Don't make the folder, just check if it is there
                    if not os.path.exists(dest_dir):
                        print(f"  [Error] Destination folder '{dest_dir}' does not exist. Skipping.")
                        continue
                    dest_path = os.path.join(dest_dir, filename)
                    
                    # Read the file content
                    data = zf.read(info.filename)
                    
                    # Handle duplicates
                    if task_id in copied_files:
                        prev_zip, prev_size, prev_path = copied_files[task_id]
                        if len(data) == prev_size:
                            print(f"  [Duplicate] '{filename}' is identical to the one already extracted from '{prev_zip}'.")
                        else:
                            print(f"  [Conflict] '{filename}' (size {len(data)} B) differs from the one in '{prev_zip}' (size {prev_size} B). Overwriting with latest.")
                    
                    # Check if the filename is already in the destination folder
                    if os.path.exists(dest_path):
                        record = (filename, folder_name)
                        if record not in overwritten_files:
                            overwritten_files.append(record)
                    
                    # Write file to target destination
                    with open(dest_path, 'wb') as f:
                        f.write(data)
                        
                    copied_files[task_id] = (zip_name, len(data), dest_path)
                    print(f"  -> Extracted to: submission_onnx/{folder_name}/{filename} ({len(data)} bytes)")
                    
        except Exception as e:
            print(f"  [Error] Failed to process {zip_name}: {e}")
            
    print("-" * 60)
    if overwritten_files:
        print(f"Overwritten {len(overwritten_files)} files already present in destination directories:")
        for filename, folder_name in sorted(overwritten_files):
            print(f"  - {folder_name}/{filename}")
        print("-" * 60)
    print(f"Done! Successfully unzipped and routed {len(copied_files)} ONNX file(s) into submission_onnx/ subdirectories.")

if __name__ == "__main__":
    main()
