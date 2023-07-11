import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from parts import Conv, FC


# 定义一个简单的网络类
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv(1, 8)
        self.conv2 = Conv(8, 16)
        self.conv3 = Conv(16, 32)
        self.fc = FC()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.fc(x3)
        return x4

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 指定网络
    net = Net()
    net.to(device=device)
    # 指定训练集地址，开始训练
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    dataset = torchvision.datasets.ImageFolder(root='../dataset', transform=transform)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for img, label in train_loader:
        # 将数据拷贝到device中
        img = img.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        break
    print(img.shape, label)
    # 使用网络参数，输出预测结果
    out = net(img)
    print(out.shape)
