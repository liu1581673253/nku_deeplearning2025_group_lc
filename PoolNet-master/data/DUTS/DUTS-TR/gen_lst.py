import os

# 子目录名称
image_dir = 'DUTS-TR-Image'
mask_dir = 'DUTS-TR-Mask'
output_file = 'train_pair.lst'

# 读取两个目录下的文件名
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

# 将文件名去除扩展名后作为键进行匹配
image_dict = {os.path.splitext(f)[0]: f for f in image_files}
mask_dict = {os.path.splitext(f)[0]: f for f in mask_files}

# 取交集作为有效配对的文件名
common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

# 写入配对信息到train_pair.lst
with open(output_file, 'w') as f:
    for key in common_keys:
        img_path = os.path.join(image_dir, image_dict[key])
        mask_path = os.path.join(mask_dir, mask_dict[key])
        f.write(f'{img_path} {mask_path}\n')

print(f'配对完成，共写入 {len(common_keys)} 行到 {output_file}')