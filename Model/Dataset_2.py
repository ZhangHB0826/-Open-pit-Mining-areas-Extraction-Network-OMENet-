import os
import re
import torch
from torch.utils.data import Dataset
import numpy as np
from osgeo import gdal
from torchvision import transforms as T
import random

class RandomAugmentation:
    """自定义数据增强类，同时处理图像和标签"""
    def __init__(self):
        self.transform = T.Compose([
            T.Lambda(lambda x: self.random_flip(x)),
            T.Lambda(lambda x: self.random_rotate(x)),
        ])

    @staticmethod
    def random_flip(data):
        image, label = data
        if random.random() < 0.5:
            image = np.flip(image, axis=2).copy()  # 水平翻转（宽度轴）
            label = np.flip(label, axis=1).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()  # 垂直翻转（高度轴）
            label = np.flip(label, axis=0).copy()
        return image, label

    @staticmethod
    def random_rotate(data):
        image, label = data
        k = random.randint(0, 3)
        image = np.rot90(image, k, axes=(1, 2)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        return image, label

class MyDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=False):  # 修改参数
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))
        self.augment = RandomAugmentation() if transform else None  # 自定义增强

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        # 读取原始数据
        image_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.labels[index])

        image = gdal.Open(image_path)
        label = gdal.Open(label_path)

        # 读取图像和标签
        image_array = np.array([image.GetRasterBand(i+1).ReadAsArray() for i in range(image.RasterCount)])
        label_array = label.GetRasterBand(1).ReadAsArray()

        # 检查形状
        if image_array.shape[0] != 5:
            raise ValueError(f"图像{self.images[index]}通道数错误")
        if label_array.ndim != 2:
            raise ValueError(f"标签{self.labels[index]}形状错误")

        # 应用数据增强（如果启用）
        if self.augment:  # 此处self.transform是RandomAugmentation实例或None
            image_array, label_array = self.augment.transform((image_array, label_array))

        # 分割为两个分支
        RGB_input = image_array[:3, :, :]   # 前3个波段
        B4B5_input = image_array[3:5, :, :] # 第4、5波段

        # 转换为张量
        RGB_tensor = torch.from_numpy(RGB_input).float()
        B4B5_tensor = torch.from_numpy(B4B5_input).float()
        label_tensor = torch.from_numpy(label_array).long()

        return (RGB_tensor, B4B5_tensor), label_tensor

        # return (B4B5_tensor, RGB_tensor), label_tensor