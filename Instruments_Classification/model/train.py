from model import Net
from torch import optim
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision


def train_net(net, device, epochs=700, batch_size=32, lr=0.00001):
    # 加载训练集
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    dataset = torchvision.datasets.ImageFolder(root='../dataset', transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 定义Adam算法
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # 交叉熵 loss
    criterion = torch.nn.CrossEntropyLoss()  # 相当于LogSoftmax+NLLLoss
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for img, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            img = img.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            # 使用网络参数，输出预测结果
            out = net(img)
            # 计算Loss
            loss = criterion(out, label)
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'kaiguan.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
        print(f'epoch:{epoch}, loss:{loss.item()}')
    print(f'best_loss:{best_loss.item()}')


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图像单通道1，分类为3。
    net = Net()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    train_net(net, device)