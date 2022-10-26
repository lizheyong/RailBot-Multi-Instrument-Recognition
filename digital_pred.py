import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import crnn
import params

def digital_pred(model_path, image_path):
    # 网络初始化
    nclass = len(params.alphabet) + 1
    model = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)
    # 加载模型
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 创建转换器，测试阶段用于将ctc生成的路径转换成最终序列，使用英文字典时忽略大小写
    converter = utils.strLabelConverter(params.alphabet)
    # 图像大小转换器
    transformer = dataset.resizeNormalize((100, 32))
    """
    读取图片,处理图片，变为二值化，缩小到100x32
    """
    image = Image.open(image_path).convert('L')
    image = transformer(image) # 读取并转换图像大小为100 x 32    w x h
    image = image.view(1, *image.size())  # (b, c, h, w) (1, 1, 32, 100)
    image = Variable(image)
    preds = model(image)  # (w c nclass)(26, 1, 37) 26为ctc生成路径长度也是传入rnn的时间步长，1是batchsize，37是字符类别数
    _, preds = preds.max(2)  # 取可能性最大的indecis size (26, 1)
    preds = preds.transpose(1, 0).contiguous().view(-1)  # 转化以为索引列表 26个元素的列表
    # 转成字符序列
    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return sim_pred

if __name__ == "__main__":
    model_path = r"pth/BlueRed_netCRNN_469_80.pth"
    image_path = r"test_red_blue/152.jpg"
    pred = digital_pred(model_path, image_path)
    print(pred)