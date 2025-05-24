import os

# 子目录名称
image_dir = 'DUTS-TR-Image'
edge_dir = "DUTS-TR-Contour"

output_file = 'edges.lst'

# 读取两个目录下的文件名
image_files = sorted([f for f in os.listdir(edge_dir) if f.endswith('.png')])

with (open(output_file, 'w') as f):
    for key in image_files:
        image=key[:-3]
        image+="jpg"
        to_print = os.path.join(image_dir,image)
        edge=os.path.join(edge_dir,key)

        f.write(f'{to_print} {edge}\n')

