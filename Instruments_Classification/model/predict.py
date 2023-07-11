import numpy as np
import torch
from model import Net
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
plt.rcParams['font.size'] = 12  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


np.set_printoptions(threshold=np.inf)
torch.set_printoptions(precision=2, threshold=float('inf'), sci_mode=False)
# plt.ion()

def pred_net(net, device,  batch_size=10):
    # 加载训练集
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((100, 100)),
        transforms.Normalize(0.5, 0.5)
    ])
    dataset = torchvision.datasets.ImageFolder(root='../testdata', transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    net.load_state_dict(torch.load('kaiguan.pth', map_location=device))  # 加载模型参数
    net.eval()

    dic = {0:'正下', 1:'左下', 2:'右下'}

    for img, label in train_loader:
        # 将数据拷贝到device中
        img = img.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)
        print(f'label:{label}')
        # 使用网络参数，输出预测结果
        out = net(img)
        # 计算loss
        loss = criterion(out, label)
        pred = torch.max(out, dim=1)
        print(f'pred:{pred.indices}')
        print(f'loss:{loss}')
        print(f'out:{out}')
        for i in range(img.size()[0]):
            to_img = transforms.ToPILImage()
            a = to_img(img[i]) #0,255
            plt.imshow(a)
            plt.title(f'分类为：{dic[pred.indices[i].item()]}！！！')
            plt.show()
        break


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    pred_net(net, device)
