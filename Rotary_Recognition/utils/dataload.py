import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.set_printoptions(profile="full")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        #ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，
        #其将每一个数值归一化到[0, 1]，其归一化方法比较简单，直接除以255即可
        transforms.Resize((100, 100)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    """
    torchvison.datasets.ImageFolder用法
    root: 数据集的文件夹路径，下面的样本分类放在不同文件夹，文件夹名为class，它自己会给每个类分indx
    transform: 就是转换，预处理下图片，旋转啊，裁剪，resize，我这里就需要用到ToTensor就行了
    """
    dataset = torchvision.datasets.ImageFolder(root='../dataset', transform=transform)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    # 加载的dataset，有class_to_idx方法，返回字典，classes方法，返回类列表
    print(dataset.class_to_idx,'\n', dataset.classes)
    # 有imgs方法:图片的路径和对应的label(元组形式), imgs的长度
    print(dataset.imgs[0],'\n', len(dataset.imgs))
    # 没有任何的transform，所以返回的还是PIL Image对象
    # print(dataset[0][1])# 第一维是第几张图，第二维为1返回label
    # print(dataset[0][0]) # 为0返回图片数据
    print(dataset[0][0].type(), dataset[0][0].size())
    for img, label in train_loader:
        print(img.shape, label)
        break

    # to_img = transforms.ToPILImage()
    # a = to_img(dataset[0][0]) #0,255
    # plt.imshow(a)
    # plt.show()
