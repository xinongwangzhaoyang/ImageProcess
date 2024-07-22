import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 定义图片和Excel路径
image_folder = r'./你的文件路径'  # 此处为存放图片的文件夹路径
excel_path = './你的文件.xlsx'   # 此处为Excel文件

# 读取Excel文件
df = pd.read_excel(excel_path)

# 准备存储图片和目标值的列表
images = []
targets = []

# 读取图片和对应的目标值
for index, row in df.iterrows():
    image_path = os.path.join(image_folder, row['Filename'])     # 改成自己文件列名，这一列的内容为图片的名称，带扩展名的图片
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is not None:
        # 缩放图片到256x256
        image = cv2.resize(image, (256, 256))
        images.append(image)
        targets.append(row['SOM'])

# 将图片和目标值转换为NumPy数组
images = np.array(images)
targets = np.array(targets)

# 对图片进行归一化处理
images = images / 255.0

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42)

# 将数据转换为PyTorch的张量
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class CustomDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

# 创建训练集和测试集的数据加载器
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  # Assuming input images are 256x256
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleCNN()

# 将模型移动到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 使用测试集进行预测
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        predictions.append(outputs.cpu().numpy())
        actuals.append(targets.cpu().numpy())

predictions = np.vstack(predictions)
actuals = np.vstack(actuals)

# 计算R²
r2 = r2_score(actuals, predictions)

# 计算RMSE
rmse = np.sqrt(mean_squared_error(actuals, predictions))

print(f'R²: {r2}')
print(f'RMSE: {rmse}')
