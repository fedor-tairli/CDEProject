import os
import json

def count_lines_in_file(file_path):
    try:
        if file_path.endswith(".py"):
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        elif file_path.endswith(".ipynb"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return sum(len(cell["source"]) for cell in data.get("cells", []) if cell["cell_type"] == "code")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def count_total_lines(root_folder):
    total_lines = 0
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith((".py", ".ipynb")):
                total_lines += count_lines_in_file(os.path.join(dirpath, filename))
    return total_lines

root_folder = "./"
print("Total lines of code:", count_total_lines(root_folder))
