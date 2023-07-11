import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from Myprocess.rotary.utils.parts import Conv, FC,FCDilas


# 定义一个简单的网络类
class SwitchDirectionClassifier(nn.Module):

    def __init__(self, num_classes=8):
        super(SwitchDirectionClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

''
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 指定网络
    net = SwitchDirectionClassifier()
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
