import os
import shutil

def copy_files(src_dir, dest_dir):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py") or file.endswith(".ipynb"):
                # Construct full file paths
                src_file = os.path.join(root, file)
                relative_path = os.path.relpath(root, src_dir)
                dest_folder = os.path.join(dest_dir, relative_path)
                
                # Create destination folder if it doesn't exist
                os.makedirs(dest_folder, exist_ok=True)
                
                # Copy the file
                shutil.copy(src_file, dest_folder)
                print(f"Copied: {src_file} -> {dest_folder}")

# Example usage
source_directory = "/remote/tychodata/ftairli/work/Projects"
destination_directory = "/remote/tychodata/ftairli/work/CDEs/References/Previous_Code"
copy_files(source_directory, destination_directory)
