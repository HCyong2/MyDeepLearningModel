"""
Bird_Classification -

Author:HCyong2
Date:2024/5/20
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import time
from torch.utils.data import DataLoader
from DataLoad import CUB
from torch.utils.tensorboard import SummaryWriter

# 设置超参数
LearningRate = 0.0008
TuningRate = 0.00015
BatchSize = 64
TotalEpoch = 20

Pretrained = False

# 模型权重保存位置
Weights_PATH = f"NO_Pretrained.pth"
# Tensorboard文件保存位置
tensorboard_PATH = f"NO_Pretrained"
# CUB_200_2011数据集所在位置
data_root = './CUB_DATA/CUB_200_2011'


print("NetWork LOADING ... ...")
# 加载预训练的ResNet-18模型并修改输出层
model = models.resnet18(pretrained=Pretrained)
num_classes = 200
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 使用GPU进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
fc_params = list(model.fc.parameters())
other_params = [param for name, param in model.named_parameters() if "fc" not in name]
optimizer = torch.optim.Adam([
    {'params': other_params, 'lr':  TuningRate},
    {'params': fc_params, 'lr': LearningRate}
])
print("----------------------NETWORK LOAD SUCCESS----------------------\n")

# 数据预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 加载数据集
time01 = time.time()
print("DATA LOADING ... ...")
train_dataset = CUB(root=data_root, is_train=True, transform=transform_train, )
test_dataset = CUB(root=data_root, is_train=False, transform=transform_test, )
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BatchSize, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)
time02 = time.time()
print("TrainSet = ", len(train_dataset))
print("TrainLoader = ", len(train_loader))
print("TestSet = ", len(test_dataset))
print("TestLoader = ", len(test_loader))
print("Load Time = ",time02-time01)
print("-----------------------DATA LOAD SUCCESS------------------------\n")

# 训练模型
print("Training ... ...")
time01 = time.time()
writer = SummaryWriter(tensorboard_PATH)
for epoch in range(TotalEpoch):
    model.train()
    training_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
    # 将训练集上的损失写入TensorBoard，并打印
    training_loss /= len(train_loader)
    writer.add_scalar('Train/Loss', training_loss, epoch + 1)
    print(f'Epoch [{epoch + 1}/{TotalEpoch}], Loss: {training_loss:.4}')

    # 验证模型(每个epoch都验证)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 将测试集上的损失和准确率写入TensorBoard，并打印
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    writer.add_scalar('Test/Loss', test_loss, epoch + 1)
    writer.add_scalar('Test/Accuracy', accuracy, epoch + 1)

    print(f'               Test Loss: {test_loss:.4}')
    print(f'               Test Accuracy: {accuracy}%')

time02 = time.time()
print("Train Time = ",time02-time01)
print("---------------------TRAINING ACCOMPLISHED----------------------\n")

# 保存模型权重
torch.save(model, Weights_PATH)
# 关闭SummaryWriter
writer.close()
