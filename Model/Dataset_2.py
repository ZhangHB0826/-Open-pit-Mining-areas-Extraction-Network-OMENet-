import os
import re
import torch
from torch.utils.data import Dataset
import numpy as np
from osgeo import gdal
from torchvision import transforms as T
import random

class RandomAugmentation:
    def __init__(self):
        self.transform = T.Compose([
            T.Lambda(lambda x: self.random_flip(x)),
            T.Lambda(lambda x: self.random_rotate(x)),
        ])

    @staticmethod
    def random_flip(data):
        image, label = data
        if random.random() < 0.5:
            image = np.flip(image, axis=2).copy()  
            label = np.flip(label, axis=1).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy() 
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
    def __init__(self, image_dir, label_dir, transform=False):  
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))
        self.augment = RandomAugmentation() if transform else None 

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.labels[index])

        image = gdal.Open(image_path)
        label = gdal.Open(label_path)

        image_array = np.array([image.GetRasterBand(i+1).ReadAsArray() for i in range(image.RasterCount)])
        label_array = label.GetRasterBand(1).ReadAsArray()

        if image_array.shape[0] != 5:
            raise ValueError(f"图像{self.images[index]}通道数错误")
        if label_array.ndim != 2:
            raise ValueError(f"标签{self.labels[index]}形状错误")

        if self.augment: 
            image_array, label_array = self.augment.transform((image_array, label_array))

        RGB_input = image_array[:3, :, :]  
        B4B5_input = image_array[3:5, :, :] 

        RGB_tensor = torch.from_numpy(RGB_input).float()
        B4B5_tensor = torch.from_numpy(B4B5_input).float()
        label_tensor = torch.from_numpy(label_array).long()

        return (RGB_tensor, B4B5_tensor), label_tensor


        # return (B4B5_tensor, RGB_tensor), label_tensor
