import os

# Paths to your folders
source_folder = ''
target_folder = ''




# Get all files in the similar folder
files_in_similar_folder = os.listdir(source_folder)

# Loop through all files in the folder
for file_name in os.listdir(source_folder):
    # Check if the file contains "(1)"
    if "(1)" in file_name:
        # Construct the old file path
        old_file_path = os.path.join(source_folder, file_name)
        
        # Construct the new file name by removing "(1)"
        new_file_name = file_name.replace("(1)", "")
        
        # Construct the new file path
        new_file_path = os.path.join(source_folder, new_file_name)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {file_name} -> {new_file_name}")

# Remove any extra space before the '.bmp' extension
valid_files = [file.replace(' .bmp', '.bmp').strip() for file in os.listdir(source_folder) if "BAD" not in file]

# Print the valid files list for checking
print("Valid files:", valid_files)


# Loop through the files in the similar folder and delete if not in the valid list
for file in files_in_similar_folder:
    if file not in valid_files:
        os.remove(os.path.join(target_folder, file))
        print(f"Deleted {file}")