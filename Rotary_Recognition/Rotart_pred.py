import torch
from Myprocess.rotary.utils.model import SwitchDirectionClassifier
import torchvision.transforms as transforms
from PIL import Image
import json

# 计算分类器的置信度
def compute_confidence(out):
    probs = F.softmax(out, dim=1)
    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
    max_probs = sorted_probs[:, 0]
    second_max_probs = sorted_probs[:, 1]
    conf = max_probs - second_max_probs
    return conf

def classification_net(net, device, img_path, model_path):
    file = open('./train_save/rotary/dic.json', 'r', encoding='gb2312')
    data = json.loads(file.read())
    file.close()

    net.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型参数
    net.eval()

    img = Image.open(img_path)
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((100, 100)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(device=device, dtype=torch.float32)
    out = net(img)
    # pred = torch.max(out, dim=1).indices[0].item()
	# 输出分类结果和置信度
    conf = compute_confidence(out)
    pred = torch.argmax(out, dim=1)
    print(f"Classification result: {pred}, Confidence: {conf}")
    if conf < 0.5:
        # 输出警告信息
        print("Warning: Low confidence in classification result!")
    return data[str(pred)]

def rotary_pred(img_path):
    model_path = r'./train_save/rotary/rotary.pth'

    device = torch.device('cpu')
    net = SwitchDirectionClassifier()
    net.to(device=device)
    name = classification_net(net, device, img_path, model_path)
    return name


if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    net = Net()
    net.to(device=device)
    img_path = r'0_6.jpg'
    model_path = r'kaiguan.pth'
    classification_net(net, device, img_path, model_path)
