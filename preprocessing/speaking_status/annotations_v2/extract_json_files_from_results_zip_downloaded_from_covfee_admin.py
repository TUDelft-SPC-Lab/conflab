# This script processes the zip file downloaded from covfee admin panel and prepares a cleaner version
# of the files for further preprocessing.
import os
import zipfile
import json
import shutil
import glob
import tempfile

#################33########### MANUAL CONFIGURATION ##############################
# This is the file directly downloaded from covfee after "loading" a database file. 
# Reference commits:
# - https://github.com/TUDelft-SPC-Lab/covfee/commit/84cd94131b911533de1e7eea742436eb99aedaa7
# - https://gitlab.ewi.tudelft.nl/edortaperez/conflab-covfee-deployment/-/tree/dbd75b1297675990981a41a468dbd70b8bf48474
home_dir = os.path.expanduser("~")
COVFEE_ADMIN_PANEL_DOWNLOADED_RESULT_ZIP_FILE_PATH: str = os.path.join(home_dir, "Downloads", "results.zip")
OUTPUT_FOLDER_PATH_FOR_JSON_FILES: str = os.path.join(os.path.dirname(__file__), "json_files")

################################################################################
# Execution below - do not modify

if not os.path.exists(COVFEE_ADMIN_PANEL_DOWNLOADED_RESULT_ZIP_FILE_PATH):
    print(f"The results zip file does not exist: {COVFEE_ADMIN_PANEL_DOWNLOADED_RESULT_ZIP_FILE_PATH}")
    exit()

print("Preparing tmp and target folders...")
tmp_folder = tempfile.mkdtemp()
os.makedirs(OUTPUT_FOLDER_PATH_FOR_JSON_FILES, exist_ok=True)
os.makedirs(tmp_folder, exist_ok=True)

print("Unzipping...")
# Unzip the file
with zipfile.ZipFile(COVFEE_ADMIN_PANEL_DOWNLOADED_RESULT_ZIP_FILE_PATH, 'r') as zip_ref:
    zip_ref.extractall(tmp_folder)


# Iterate over all json files in the tmp_folder
for json_file in glob.glob(os.path.join(tmp_folder, '*.json')):
    # Open the json file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract the global_unique_id
    global_unique_id = data.get('global_unique_id', None)

    if global_unique_id:
        # Rename the file to "{global_unique_id}.json"
        target_file_name = f"{global_unique_id}.json"
        print(f"Moving {target_file_name}")
        target_file_path = os.path.join(OUTPUT_FOLDER_PATH_FOR_JSON_FILES, target_file_name)
        
        # Delete target_file_name if it exists before moving
        if os.path.exists(target_file_path):
            os.remove(target_file_path)

        os.rename(json_file, target_file_path)
    else:
        print(f"Skipping {os.path.basename(json_file)}: global_unique_id not found.")

print("Cleanup...")
shutil.rmtree(tmp_folder)