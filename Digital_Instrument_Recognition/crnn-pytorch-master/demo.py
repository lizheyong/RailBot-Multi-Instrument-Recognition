import torch
from torch.autograd import Variable
import utils
import dataset
import os
import glob
from PIL import Image, ImageOps
import numpy as np

import models.crnn as crnn
import params
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--model_path', type = str, required = True, help = 'crnn model path')
# parser.add_argument('-i', '--image_path', type = str, required = True, help = 'demo image path')
args = parser.parse_args()

# model_path = args.model_path
# image_path = args.image_path

model_path = r"F:\share\数字训练\model\5805ALL_1.pth"
# image_path = 'D:/LiZheyong/crnn-pytorch-master/7/P00027_16537415395209_0_224.jpg'

# 网络初始化
nclass = len(params.alphabet) + 1
model = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)
if torch.cuda.is_available():
    model = model.cuda()

# 加载模型
print('loading pretrained model from %s' % model_path)
if params.multi_gpu:
    model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_path))
model.eval()
# 创建转换器，测试阶段用于将ctc生成的路径转换成最终序列，使用英文字典时忽略大小写
converter = utils.strLabelConverter(params.alphabet)

# 图像大小转换器
transformer = dataset.resizeNormalize((100, 32))

"""
读取测试数据文件夹下的图片到txt
"""
root_path = r"C:\Users\zheyong\Desktop\a"
paths = glob.glob(os.path.join(root_path, '*.jpg'))

# with open(root_path + r"\test.txt","w") as w:
#     for path in paths:
#         w.write(path+"\n")
"""
依次处理这些图片，变为二值化，缩小到
"""

# input_file = input_path + r'\test.txt'
# with open(input_file, 'r') as f:
#     data = f.read().splitlines()
data = paths

for i in range(len(data)):
    image = Image.open(data[i])
    image_copy = image
    image = image.convert('L')
    image = transformer(image) # 读取并转换图像大小为100 x 32    w x h
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())  # (b, c, h, w) (1, 1, 32, 100)
    image = Variable(image)
    preds = model(image)  # (w c nclass)(26, 1, 37) 26为ctc生成路径长度也是传入rnn的时间步长，1是batchsize，37是字符类别数

    _, preds = preds.max(2)  # 取可能性最大的indecis size (26, 1)
    preds = preds.transpose(1, 0).contiguous().view(-1)  # 转化以为索引列表 26个元素的列表
    # 转成字符序列
    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))

    save_root = root_path + r"\result\ "  # 保存地址
    path = save_root + str(sim_pred) + '_' + str(i)  +'.jpg'  # 保存地址
    try:
        image_copy.save(path, quality=95)
        # print('图片保存成功，保存在' + save_root + "\n")
    except:
        print('图片保存失败')






