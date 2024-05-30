# 在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3

## 介绍

使用 mmdetection 架构在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3。

本仓库提供了训练配置文件和最终的模型权重文件。

### 模型

**Faster R-CNN：**

Faster R-CNN 是一种经典的两阶段目标检测模型，整体架构由卷积特征提取器（ResNet-50）、区域建议网络（RPN）、Rol池化层、分类回归网络四个部分组成。

**YOLO V3：**

YOLO（You Only Look Once）是一种经典的单阶段目标检测模型，主要由骨干网络（Darknet-53）、特征金字塔网络（FPN）和检测头三部分组成。

**数据集**

VOC 数据集中包含20个物体类别，这些类别涵盖了常见的物体，如动物、交通工具和日常用品。

## 使用说明

```bash
data
  |__ VOC2012                 # VOC2012数据库
        |__ Annotations       # 我仓库里并没有完整的数据库
        |__ ImageSets         # 只是表明正确的文件路径应该是这样
        |__ JPEGImages
        |__ VOC_build.py      # 用于分割数据集
my_code
  |__ faster-rcnn.py          # faster-rcnn 的训练配置文件
  |__ yolo_v3.py              # yolo_v3 的训练配置文件
  |__ test_one_picture.py     # 画一张图片的检测结果
  |__ draw_proposal.py        # 画faster-rcnn的第一阶段预测框
  |__ final_faster-rcnn       # faster-rcnn训练日志
  |__ final_yolo              # yolo v3训练日志
  |__ image_result            # 图片检测结果
  |__ result                  # 训练可视化图片
  |__ test_images             # 一些测试图片
  
README.md                     # 使用文档（即本文档）
实验报告.pdf                   # 我写的实验报告（老费劲了）
```

**使用前准备**

请在python虚拟环境中配置以下package

matplotlib              3.9.0
mmcv                    2.0.0
mmdet                   3.3.0
mmengine             0.10.4
numpy                   1.26.4
torch                      2.0.1+cu117
torchaudio             2.0.2+cu118
torchvision             0.15.2+cu117

之后将 data、my_code 文件夹复制到 mmdetection 中
如果想训练模型，请根据data文件的路径正确加载VOC数据集

```
cd mmdetection
```

**训练模型**

```shell
python tools\train.py my_code\faster-rcnn.py --work-dir my_code
```

**测试模型**

```shell
python tools\test.py my_code\faster-rcnn.py my_code\epoch_1.pth --out epoch_1.pkl
```

**在指定图片上测试并生成结果**

```shell
python my_code/test_one_picture.py my_code/test_images/2007_000170.jpg

python my_code/test_one_picture.py --config my_code/yolo_v3.py --checkpoint my_code/my_model/yolo_v3/my_code/epoch_47.pth my_code/test_images/cats.jpg

python my_code/test_one_picture.py --config my_code/faster-rcnn.py --checkpoint my_code/my_model/faster-rcnn/epoch_9.pth my_code/test_images/bird.jpg
```

**绘制proposal框**

```shell
python my_code/draw_proposal.py my_code/test_images/2010_000922.jpg
```

**训练过程可视化**（绘制 loss图、mAP图）

```shell
python tools/analysis_tools/analyze_logs.py plot_curve my_code/train_log/vis_data/20240524_143431.json --keys loss_cls --legend loss_cls --out my_code/

python tools/analysis_tools/analyze_logs.py plot_curve my_code/final_yolo/vis_data/20240527_191322.json --keys loss loss_conf --legend loss loss_conf --out my_code/result/yolo_loss

python tools/analysis_tools/analyze_logs.py plot_curve my_code/final_yolo/vis_data/20240527_191322.json --keys loss_cls --legend loss_cls --out my_code/result/yolo_loss_cls

```



## 参考资料

[使用mmdetection测试以及生成检测结果 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/166144827)

[【MMDetection-学习记录】 训练自己的VOC数据集_mmdetection在多 gpu 上训练-CSDN博客](https://blog.csdn.net/qq_41251963/article/details/112940253?spm=1001.2014.3001.5506)