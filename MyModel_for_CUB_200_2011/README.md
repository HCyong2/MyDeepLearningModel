# 微调ResNet-18实现鸟类识别

## 介绍

### 模型

使用了在 ImageNet 上预训练的 ResNet-18 模型，将最后一层输出层大小修改为 200并随机初始化。之后通过交叉熵损失函数和Adam优化器进行反向传播训练。其中输出层学习率较大，其他层学习率小进行微调。

### 数据集

使用 CUB-200-2011 数据集，其包含了 200 种鸟类图片，每种50多张，共11788张图片。训练集有5994张，测试集有5794张。

## 使用说明

```bash
python_code                   # 训练时使用的python代码
  |__ Bird_Classification.py     # 神经网络训练程序
  |__ Dataload.py                # 加载CUB-200-2011数据集
  |__ ModelLoad.py               # 加载已经训练好的网络

my_model                      # 训练好的模型权重
  |__ BEST_MODEL.pth             # 我训练出来的最佳模型
  |__ NO_Pretrained.pth          # 没有预训练的模型

experiment                    # 调整参数时保存的tensorboard文件
pictures                      # 文档图片
README.md                     # 使用文档（即本文档）
```

## 训练神经网络

### 1.配置python 环境

运行本仓库的py代码前，请确保环境中有以下package

```
numpy                   1.26.4
pillow                  10.3.0
torch                   2.3.0+cu121
torchaudio              2.3.0+cu121
torchvision             0.18.0+cu121
tensorboard             2.16.2
tensorboard-data-server 0.7.2
```

### 2.设置参数

打开 Bird_Classification.py 根据你的需求设置对应的参数

**data_root 必须修改为已经加压后的CUB_200_2011数据集所在位置**

```py
# 设置超参数
LearningRate = 0.0008
TuningRate = 0.00015
BatchSize = 64
TotalEpoch = 20

Pretrained = True

# 模型权重保存位置
Weights_PATH = f"LR{LearningRate}_TR{TuningRate}_BZ{BatchSize}_EP{TotalEpoch}.pth"
# Tensorboard文件保存位置
tensorboard_PATH = f"LR{LearningRate}_TR{TuningRate}_BZ{BatchSize}_EP{TotalEpoch}"
# CUB_200_2011数据集所在位置
data_root = 'Your/path/to/CUB_200_2011'
```

### 3.进行训练

正常训练结果如图
请忽略warning信息（\^w\^），能跑就行

![image](https://github.com/HCyong2/MyModel_for_CUB_200_2011/tree/master/MyModel_for_CUB_200_2011/pictures/test01.png)

### 4.模型训练过程可视化

在对应python环境中，进入Tensorboard文件保存位置，运行tensorboard命令。

```bash
(venv) PS D:\MyModel_for_CUB_200_2011\Python_code> cd .\experiment
(venv) PS D:\MyModel_for_CUB_200_2011\experiment> tensorboard --logdir LR0.001_EP10_BZ32
```

### 5.模型重新加载

打开 ModelLoad.py ，修改model文件位置并运行



## 参考资料

CUB-200-2011 数据集加载：
*https://blog.csdn.net/weixin_41735859/article/details/106937174*
