import os
import glob
from PIL import Image, ImageOps
import numpy as np
import collections

"""
读取测试数据文件夹下的图片到txt
"""
# root_path = r'D:\crnn-pytorch-master\crnn-pytorch-master\valid_data'
# paths = glob.glob(os.path.join(root_path, '*.jpg'))
# with open(r"C:\Users\zheyong\Desktop\变电站项目\mytest.txt","w") as w:
#     for path in paths:
#         w.write(path+"\n")
# print(paths)
# paths.sort()
# print(paths)

# with open(r"C:\Users\zheyong\Desktop\数字信号处理\a.txt", 'r', encoding='UTF-8') as f:
#     data = f.read().splitlines()
#



x=[]
with open(r"C:\Users\zheyong\Desktop\变电站项目\mytest.txt","r") as w:
    a = w.read().split()
    for i in a:
        c = i.split('data\\')[1].replace('.jpg','')
        x.append(c)

for i in x:
    if ' ' in i:
        print(i)




