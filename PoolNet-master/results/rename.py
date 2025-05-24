import os

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            name, ext = os.path.splitext(filename)
            new_name = name[:4] + ext
            new_path = os.path.join(folder_path, new_name)
            os.rename(full_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

# 设置目标文件夹路径
folder = 'run-1-sal-e'  # ← 修改为你的文件夹路径
rename_files(folder)