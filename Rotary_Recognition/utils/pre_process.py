import os
import glob
from PIL import Image, ImageOps
import numpy as np

"""
读取测试数据文件夹下的图片到txt
"""
root_path = '../dataset/test'
paths = glob.glob(os.path.join(root_path, '*.bmp'))
print(paths)
# paths.sort()
# print(paths)
with open("../dataset/test/mytest.txt","w") as w:
    for path in paths:
        w.write(path+"\n")
"""
依次处理这些图片，变为二值化，缩小到100x100
"""
input_file = '../dataset/test/mytest.txt'
with open(input_file, 'r') as f:
    data = f.read().splitlines()

for i in range(len(data)):
    origin_image = Image.open(data[i]) # 读取图片
    origin_image = origin_image.resize((100, 100), Image.ANTIALIAS)
    # origin_image.thumbnail((100,100),2) # 变为100x100的
    bw = origin_image.convert('L') # 转为二值图
    reverse_bw = ImageOps.invert(bw)
    save_root = '../dataset/test_process/detect_result/'  # 保存地址
    path = save_root + str(i) +'.jpeg'  # 保存地址
    try:
        reverse_bw.save(path, quality=95)
        print('图片保存成功，保存在' + save_root + "\n")
    except:
        print('图片保存失败')