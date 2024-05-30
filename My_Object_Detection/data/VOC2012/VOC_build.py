"""
VOC_build -

Author:霍畅
Date:2024/5/24
"""
import os
import random

random.seed(0)

xmlfilepath = r'./Annotations'
saveBasePath = r"./ImageSets/Main/"

trainval_percent = 0.8  # 80% 数据用于训练和验证，20% 数据用于测试
train_percent = 0.75    # 训练和验证数据中 75% 用于训练，25% 用于验证

temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

# 生成较小数据集，用于检测代码正确性
# num = len(total_xml)//200

num = len(total_xml)
print(num)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("train size", tr)
ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()