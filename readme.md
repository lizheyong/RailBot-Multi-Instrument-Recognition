* 用法

```python
from digital_pred import digital_pred

model_path = r"pth/BlueRed_netCRNN_469_80.pth"
image_path = r"test_red_blue/152.jpg"
pred = digital_pred(model_path, image_path) # 返回读数pred
```



* crnn.py 为模型文件
* alphabets为字符表，0123456789 . -
* pth文件夹下为保存的模型
* test_red_blue为测试用的蓝色和红色数字仪表截取后图片