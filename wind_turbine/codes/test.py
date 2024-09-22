# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt


# 数据加载和预处理
def load_and_preprocess_data():
    # 假设数据已经以.npy格式保存在本地
    train_images = np.load('../dataset/image/train_images_17.npy')
    train_labels = np.load('../dataset/image//train_labels_17.npy')
    # coord = np.load('coord.npy')

    # 转换为torch tensors
    train_images = torch.tensor(train_images).float()
    train_labels = torch.tensor(train_labels).float()

    # 数据集封装
    dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return train_loader


# 模型构建
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    return model


# 训练模型
def train_model(model, train_loader, epochs=50):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data = data.permute(0, 3, 1, 2)  # 调整维度顺序
            optimizer.zero_grad()
            output = model(data)
            # 确保 output 和 target 尺寸相同
            output = output.view(-1)  # 将 output 展平为一维
            loss = criterion(output, target.view(-1))  # 将 target 也展平为一维
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    return model


# 预测并显示错误预测的图像
def show_wrong_predictions(model, train_loader):
    model.eval()
    with torch.no_grad():
        wrong = []
        for data, target in train_loader:
            output = model(data)
            predicted = torch.round(torch.sigmoid(output))
            wrong.extend((i for i, x in enumerate(predicted.view(-1)) if x.item() != target[i].item()))

        count = 1
        for i in wrong:
            img = data[i].numpy().transpose(1, 2, 0)
            plt.subplot(3, 3, count)
            plt.imshow(img)
            count += 1
            if count > 9:
                break


# plt.show()


# 主函数
def main():
    train_loader = load_and_preprocess_data()
    model = build_model()
    model = train_model(model, train_loader)
    show_wrong_predictions(model, train_loader)
    torch.save(model.state_dict(), 'resnet50_model.pth')


if __name__ == "__main__":
    main()
