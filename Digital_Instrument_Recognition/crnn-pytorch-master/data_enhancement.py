from PIL import Image
import os

# 指定文件夹路径
folder_path = r"C:\Users\zheyong\Desktop\gray"

# 读取文件夹中的所有图像
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        # 解析图像标签和序号
        label, number = filename.split('_')

        # 加载原始图像
        img = Image.open(os.path.join(folder_path, filename))

        # 上下左右裁剪3行（列）个像素
        img_crop_up = img.crop((0, 3, img.width, img.height - 3))
        img_crop_down = img.crop((0, 0, img.width, img.height - 6))
        img_crop_left = img.crop((3, 0, img.width - 3, img.height))
        img_crop_right = img.crop((0, 0, img.width - 6, img.height))

        # 顺时针、逆时针旋转2度
        img_rotate_cw = img.rotate(2)
        img_rotate_ccw = img.rotate(-2)

        # 保存数据增强后的图像
        img_crop_up.save(fr"{folder_path}\{label}_0001{number}.jpg")
        img_crop_down.save(fr"{folder_path}\{label}_0002{number}111.jpg")
        img_crop_left.save(fr"{folder_path}\{label}_0003{number}.jpg")
        img_crop_right.save(fr"{folder_path}\{label}_0004{number}.jpg")
        img_rotate_cw.save(fr"{folder_path}\{label}_0005{number}.jpg")
        img_rotate_ccw.save(fr"{folder_path}\{label}_0006{number}.jpg")
