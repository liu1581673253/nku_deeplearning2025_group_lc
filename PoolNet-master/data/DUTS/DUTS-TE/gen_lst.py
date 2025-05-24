import os

# 子目录名称
image_dir = 'Image'

output_file = 'test.lst'

# 读取两个目录下的文件名
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])


# 写入配对信息到train_pair.lst
with open(output_file, 'w') as f:
    for key in image_files:
        img_path = os.path.join(image_dir, key)
        f.write(f'{key}\n')

