"""
ModelLoad - Load model and calculate accuracy.

Author:HCyong2
Date:2024/5/20
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from DataLoad import CUB

# 加载模型
PATH = r".\BEST_MODEL.pth"
model = torch.load(PATH)
model.eval()

# 数据预处理


transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 加载数据集
print("DATA LOADING ... ...")
test_dataset = CUB(root='./CUB_DATA/CUB_200_2011', is_train=False, transform=transform_test, )
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
print("TestSet = ", len(test_dataset))
print("TestLoader = ", len(test_loader))

# 计算准确率
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
correct = 0
total = 0
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy}%')