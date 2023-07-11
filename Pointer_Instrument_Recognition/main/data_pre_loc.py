import os
import json
import numpy as np
import math
import matplotlib.pyplot as plt
from config_loc import config_loc as cfg
from scipy.ndimage import gaussian_filter
import cv2
import PIL
from torchvision import transforms
import torch
from PIL import Image, ImageFont, ImageDraw
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# json变成加入高斯的np
def json_to_numpy(dataset_path):
    with open(dataset_path,encoding='UTF-8') as fp:
        json_data = json.load(fp)
        points = json_data['shapes']

    # print(points)
    landmarks = []
    for point in points:
        for p in point['points']:
            landmarks.append(p)

    # print(landmarks)
    landmarks = np.array(landmarks)
    landmarks = landmarks.reshape(-1, 2)

    # 保存为np
    # np.save(os.path.join(save_path, name.split('.')[0] + '.npy'), landmarks)

    return landmarks


def generate_heatmaps(landmarks, h, w, height, width, sigma):

    heatmaps = []
    for points in landmarks:
        heatmap = np.zeros((height, width))

        ch = int(height * points[1] / h)
        cw = int(width * points[0] / w)
        heatmap[ch][cw] = 1

        heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
        am = np.amax(heatmap)
        heatmap /= am / 255
        heatmaps.append(heatmap)

    heatmaps = np.array(heatmaps)
    # heatmaps = np.expand_dims(heatmaps, axis=0)

    return [heatmaps, h, w, height, width]


def show_heatmap(heatmaps):
    heatmaps = heatmaps[0]
    for heatmap in heatmaps:
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.show()


def heatmap_to_point(heatmaps,img_w,img_h,input_w,input_h):
    # img_h = heatmaps[1]
    # img_w = heatmaps[2]
    # input_h = heatmaps[3]
    # input_w = heatmaps[4]
    # heatmaps = heatmaps[0]

    points = []
    for heatmap in heatmaps:
        pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        point0 = img_w * (pos[0] / input_w)
        point1 = img_h * (pos[1] / input_h)
        points.append([point1, point0])
    return np.array(points)


def show_inputImg_and_keypointLabel(imgPath, heatmaps):
    heatmaps = heatmaps[0]
    points = []
    for heatmap in heatmaps:
        pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        points.append([pos[1], pos[0]])
    img = PIL.Image.open(imgPath).convert('RGB')
    img = img.resize((512, 512),Image.ANTIALIAS)
    # img = transforms.ToTensor()(img)  # 3*512*512

    # img = img.unsqueeze(0)  # 增加一维
    # resize = torch.nn.Upsample(scale_factor=(0.25, 0.25), mode='bilinear', align_corners=True)
    # img = resize(img)

    # img = img.squeeze(0)  # 减少一维

    # print(img.shape)
    print('aaaa')

    # img = transforms.ToPILImage()(img)
    draw = ImageDraw.Draw(img)
    for point in points:
        print(point)
        draw.point((point[0], point[1]), fill='yellow')

    # 保存

    img.save(os.path.join('out.jpg'))


if __name__ == '__main__':
    landmarks = json_to_numpy('data/trainData/trainWay/labels/P502_16537941246846_0_N.json')
    print('关键点坐标', landmarks, '-------------', sep='\n')
    img = PIL.Image.open('data/trainData/trainWay/imgs/P502_16537941246846_0_N.jpg').convert('RGB')
    w=img.size[0]
    h=img.size[1]
    img = img.resize((512, 512),Image.ANTIALIAS)

    heatmaps = generate_heatmaps(landmarks, h, w, img.size[1], img.size[0], (cfg['gauss_h'], cfg['gauss_w']))
    # print(heatmaps)
    print(heatmaps[0].shape)

    # show heatmap picture
    # show_heatmap(heatmaps)

    # show cut image and the keypoints
    show_inputImg_and_keypointLabel('data/trainData/trainWay/imgs/P502_16537941246846_0_N.jpg', heatmaps)
