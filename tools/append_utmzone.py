import os
import shutil

def append_suffix_to_files_in_subfolders(parent_folder):
    # Iterate through all subfolders in the parent folder
    for subdir, dirs, files in os.walk(parent_folder):
        # Get the last 4 characters of the subfolder name
        folder_name = os.path.basename(subdir)
        suffix = folder_name[-5:]
        
        # Iterate through all files in the current subfolder
        for file_name in files:
            # Get the file name and extension
            base_name, extension = os.path.splitext(file_name)
            
            # New file name with appended suffix
            new_file_name = f"{base_name}_{suffix}{extension}"
            new_file_path = os.path.join(subdir, new_file_name)
            old_file_path = os.path.join(subdir, file_name)
            
            # Rename the file
            shutil.move(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_path}")

# Example usage:
parent_folder = r"N:\isipd\projects\p_planetdw\data\dw_detection\PlanetScope\training_data\temp"
append_suffix_to_files_in_subfolders(parent_folder)
