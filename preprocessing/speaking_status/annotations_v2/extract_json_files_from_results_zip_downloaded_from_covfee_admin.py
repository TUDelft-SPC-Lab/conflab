# This script processes the zip file downloaded from covfee admin panel and prepares a cleaner version
# of the files for further preprocessing.
import zipfile
import json
import shutil
import tempfile
from pathlib import Path

#################33########### MANUAL CONFIGURATION ##############################
# This is the file directly downloaded from covfee after "loading" a database file. 
# Reference commits:
# - https://github.com/TUDelft-SPC-Lab/covfee/commit/84cd94131b911533de1e7eea742436eb99aedaa7
# - https://gitlab.ewi.tudelft.nl/edortaperez/conflab-covfee-deployment/-/tree/dbd75b1297675990981a41a468dbd70b8bf48474
COVFEE_ADMIN_PANEL_DOWNLOADED_RESULT_ZIP_FILE_PATH: str = Path.home() / "Downloads" / "results.zip" 
OUTPUT_FOLDER_PATH_FOR_JSON_FILES: str = Path(__file__).parent / "json_files"

################################################################################
# Execution below - do not modify

if not COVFEE_ADMIN_PANEL_DOWNLOADED_RESULT_ZIP_FILE_PATH.exists():
    print(f"The results zip file does not exist: {COVFEE_ADMIN_PANEL_DOWNLOADED_RESULT_ZIP_FILE_PATH}")
    exit()

print("Preparing tmp and target folders...")
OUTPUT_FOLDER_PATH_FOR_JSON_FILES.mkdir(parents=True, exist_ok=True)
tmp_folder =  Path(tempfile.mkdtemp())
tmp_folder.mkdir(parents=True, exist_ok=True)

print("Unzipping...")
# Unzip the file
with zipfile.ZipFile(COVFEE_ADMIN_PANEL_DOWNLOADED_RESULT_ZIP_FILE_PATH, 'r') as zip_ref:
    zip_ref.extractall(tmp_folder)


# Iterate over all json files in the tmp_folder
for json_file_path in tmp_folder.glob('*.json'):
    # Open the json file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract the global_unique_id
    global_unique_id = data.get('global_unique_id', None)

    if global_unique_id:
        # Rename and move the json file to the global_unique_id.json
        print(f"Moving {global_unique_id}.json")
        target_json_file_path = OUTPUT_FOLDER_PATH_FOR_JSON_FILES / f"{global_unique_id}.json"
        
        # Delete target_file_name if it exists before moving
        target_json_file_path.unlink(missing_ok=True)
        json_file_path.rename(target_json_file_path)
    else:
        print(f"Skipping {json_file_path}: global_unique_id not found.")

print("Cleanup...")
shutil.rmtree(tmp_folder)