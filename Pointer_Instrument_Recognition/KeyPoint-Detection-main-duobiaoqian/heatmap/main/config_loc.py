import os
import torch

config_loc = {
    # 网络训练部分
    # 'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'device': torch.device("cuda"),
    'batch_size': 1,
    'epochs': 510,
    'save_epoch': 100,
    'learning_rate': 0.0001,
    'lr_scheduler': 'step1',  # 可以选择'step1','step2'梯度下降，'exponential'指数下降

    # 原图尺寸
    'img_h': 3036,
    'img_w': 4024,

    # 裁剪后的尺寸
    'cut_h': 512,
    'cut_w': 512,

    # 网络输入的图像尺寸
    'input_h': 512,
    'input_w': 512,

    # 高斯核大小
    'gauss_h': 25,
    'gauss_w': 25,

    # 关键点个数
    'kpt_n': 5,

    # 网络评估部分
    'test_batch_size': 1,
    'test_threshold': 0.5,

    'path': '/home/lwm/Disk_D/Projects/IndustrialProjects/Haier',

    # 设置路径部分
    'train_date': 'trainData',
    'train_way': 'trainWay',
    'test_date': 'testData',
    'test_way': 'testWay',

    # 调用的模型
    'pkl_file': 'min_loss.pth',

    # 是否加载预训练模型
    'use_old_pkl': False,
    'old_pkl': 'min_loss.pth',

    # # pytorch < 1.6
    # 'pytorch_version': False,

    # remember location
    'start_x': 200,
    'start_y': 200,
    'start_angle': 0,

    # max x,y
    'max_x': 300,
    'max_y': 250,
    'max_angle': 90,

    # min x,y
    'min_x': 100,
    'min_y': 100,

    # key points relative location
    'distance_12': 360,
    'distance_13': 200,
    'distance_23': 410,

    'delta': 50,

    # 'photo_to_world':
}
