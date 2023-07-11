# 变电站仪表检测与识别

我的部分数据集：

https://lizheyong.com/dataset-substation

## 预处理对各种仪表开关分类

预处理是指对现场图片直接进行目标识别。目前方案是对指示灯、开光这种含有离散状态值的目标物体，对其类别与状态直接识别出来。以指示灯为例，将红色指示灯亮、红色指示灯灭、黄色指示灯亮、黄色指示灯灭分别作为四种物体进行识别。针对指针式仪表、数字式仪表、旋钮这种含有连续状态量的物体，将其分别作为一个物体识别出来。再将候选框区域裁剪出来进行后处理，对连续状态量进行识别。

以下是针对多目标面板、异常光照等极端环境下的测试结果。最后一张为训练日志记录的验证集测试指标，其中IOU取0.5的时候，AP值可以达到0.996，可以看出其识别效果显著。但同时由于图片数据少和单一，也可能存在过拟合的现象

![image-20230711160701972](https://github.com/lijinchao98/digital_pred_func/blob/master/img/image-20230711160701972.png)

![image-20230711160836054](https://github.com/lijinchao98/digital_pred_func/blob/master/img/image-20230711160836054.png)

## 指针识别

### 1. Yolo检测关键刻度，指针头尾

#### 流程

1. Yolo在整个面板检测出指针仪表
2. 用分类网络对仪表分类，判断是哪种指针仪表
3. 再训练一个Yolo检测指针仪表的关键刻度，指针头，指针尾，头尾得到指针直线
4. 根据角度估计读数，要注意指针刻度可能不是均匀的

#### 效果

![image-20230711161727835](https://github.com/lijinchao98/digital_pred_func/blob/master/img/image-20230711161727835.png)

![image-20230711161733110](https://github.com/lijinchao98/digital_pred_func/blob/master/img/image-20230711161733110.png)

![image-20230711161746081](https://github.com/lijinchao98/digital_pred_func/blob/master/img/image-20230711161746081.png)

![image-20230711161755178](https://github.com/lijinchao98/digital_pred_func/blob/master/img/image-20230711161755178.png)

#### 局限

只适用个别的指针表，弃用

### 2. 关键点检测

参考一下工业关键点检测：

https://github.com/ExileSaber/KeyPoint-Detection/tree/main

https://blog.csdn.net/weixin_41782172/article/details/119249916

UNet，Heatmap，效果如下，这里只是部分，实际种有很多类的指针表。但是分辨率变化，模糊，倾斜，都会影响，需要丰富的数据标注训练，实用性差，弃用。

![56_7_keypoint](https://github.com/lijinchao98/digital_pred_func/blob/master/img/56_7_keypoint.jpg)

![55_1_keypoint](https://github.com/lijinchao98/digital_pred_func/blob/master/img/55_1_keypoint.jpg)

### 3. 模板匹配

还是用传统的，简单的。先分类出来哪种指针表，对每种表用之前的简单标记好的模板，提取指针就好了，效果还行。

后面实际用的时候，会存在光线变化，镜面反光，需要进行一些处理。

## 开关，灯识别

这个分类就好了

## 数字仪表识别

主要是小数点要检测出来。网上一些方法对于小数点的处理，是把小数点的位置当先验信息，比如固定就在哪一位，后期添加上去。

### 1. CRNN+CTC

前面Yolo检测出了数字区域，对数字区域识别

参考 https://blog.csdn.net/Enigma_tong/article/details/117735799

https://github.com/Holmeyoung/crnn-pytorch
值得注意的事情是，crnn训练的时候，由于我们做的是数字式仪表的读数，所以aphabet为0123456789. 把crnn中[lstm](https://so.csdn.net/so/search?q=lstm&spm=1001.2101.3001.7020)中的nclass的数值设置为12=11+1.（有负号的话再加1）

感谢这个作者当时对我的耐心回复！
![55_1_keypoint](https://github.com/lijinchao98/digital_pred_func/blob/master/img/f29fb379ada8eb77be1fd563eaf5aab.jpg)


效果是可以的。就是每次新类型的数字表，需要大量数据去加进去训练；而且对数据分布也有要求，比如训练的都是237，239这附近的数据，识别的时候缺变成872这种没有出现的数据，就会有问题。

还有第一步Yolo裁剪的时候可能不准确，影响后续识别，比如那个数字仪表倾斜了一点，采取了将裁剪区域进行上下左右轻微平移（也可以考虑轻微旋转）得到多个待识别区域，进行识别，投票，异常结果警告。

是可以将不同数字仪表的数据放一起训练的。

主要优点就是简单，标注的时候看着图输入数值就行。对于小数点的检测很好。标注的时候由于手动输入可能会有错误标签，多打了空格之类，加载数据训练会报错，写了几行小代码去检查这些。

前期一直用的这个方法。后面又换各种不同的数字仪表，不同的拍摄场景，又不能给很多训练数据，那就想其他方法。

### 2. Yolo+Yolo

对裁剪出来的待识别数字区域，再用一个yolo去检测‘1’，‘2’，...‘-‘，’.’这样。但是yolo检测小数点可能效果有限，其实也还行。

尝试过形态学小数点检测，但是这个不同仪表，分辨率阈值难定。

ps: 我看crnn就挺好。咋可能又想训练数据很少，还想用一个模型适用各种不同地方的不同仪表。。。



