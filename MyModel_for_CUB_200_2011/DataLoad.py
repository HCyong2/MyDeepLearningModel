"""
Dataload -

Author:HCyong2
Date:2024/5/20

reference：
https://blog.csdn.net/weixin_41735859/article/details/106937174
"""
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch

class CUB():
    def __init__(self, root, is_train=True, data_len=None,transform=None, target_transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        # 图片索引
        img_name_list = []
        for line in img_txt_file:
            # 最后一个字符为换行符
            img_name_list.append(line[:-1].split(' ')[-1])

        # 标签索引，每个对应的标签减１，标签值从0开始
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)

        # 设置训练集和测试集
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        train_label_list = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        test_label_list = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

        if self.is_train:
            # 使用 PIL.Image.open 读取图像，并转换为 numpy 数组
            self.train_img = [np.array(Image.open(os.path.join(self.root, 'images', train_file))) for train_file in
                              train_file_list[:data_len]]
            # 读取训练集标签
            self.train_label = train_label_list
        else:
            self.test_img = [np.array(Image.open(os.path.join(self.root, 'images', test_file))) for test_file in
                             test_file_list[:data_len]]
            self.test_label = test_label_list

    # 数据增强
    def __getitem__(self,index):
        # 训练集
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
        # 测试集
        else:
            img, target = self.test_img[index], self.test_label[index]

        if len(img.shape) == 2:
            # 灰度图像转为三通道
            img = np.stack([img]*3,2)
        # 转为 RGB 类型
        img = Image.fromarray(img,mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

if __name__ == '__main__':
    # 以pytorch中DataLoader的方式读取数据集
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    print("DATA LOADING ... ...")
    train_dataset = CUB(root='./CUB_DATA/CUB_200_2011', is_train=True, transform=transform_train, )
    test_dataset = CUB(root='./CUB_DATA/CUB_200_2011', is_train=False, transform=transform_train,)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    print("TrainSet = ",len(train_dataset))
    print("TrainLoader = ",len(train_loader))
    print("TestSet = ",len(test_dataset))
    print("TestLoader = ",len(test_loader))
    print("-----------------------DATA LOAD SUCCESS------------------------")

