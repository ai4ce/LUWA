import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd

# csv_path = "../LUA_Dataset/CSV/256_50x_6w_test.csv"
# img_path = "../LUA_Dataset/256/50x/texture"

# def StatisticsImage(train_data):
#     means = torch.zeros(3)
#     stds = torch.zeros(3)

#     for img, label in train_data:
#         means += torch.mean(img, dim=(1, 2))
#         stds += torch.std(img, dim=(1, 2))

#     means /= len(train_data)
#     stds /= len(train_data)

#     print(f'Calculated means: {means}')
#     print(f'Calculated stds: {stds}')
#     return means, stds

# def TrainTransform(means, stds):
#     train_transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomRotation(5),
#         transforms.RandomHorizontalFlip(0.5),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=means,
#                              std=stds)
#     ])
#     return train_transforms

# def TestTransform(means, stds):
#     test_transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=means,
#                              std=stds)
#     ])
#     return test_transforms




class TribologyDataset(Dataset):
    """Tribology dataset."""

    def __init__(self, csv_path, img_path):
        self.img_path = img_path
        self.csv_path = csv_path
        self.used_labels = ["ANTLER", "BEECHWOOD", "BEFOREUSE", "BONE", "IVORY",
                            "SPRUCEWOOD"]

        self.labels_maps = {"ANTLER": 0, "BEECHWOOD": 1, "BEFOREUSE": 2, "BONE": 3,
                            "IVORY": 4, "SPRUCEWOOD": 5, "BARLEY": 6, "FERN": 7, "HORSETAIL": 8}

        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.data = []

        for name, label in zip(self.image_name_all, self.labels_all):

            if label in self.used_labels:
                img_name = os.path.join(img_path, label.lower(), name)
                self.data.append([img_name, self.labels_maps[label]])

        self.data_len = len(self.data)


    def get_statistics(self):
        data = self.data
        means = torch.zeros(3)
        stds = torch.zeros(3)

        for img, _ in data:
            actual_img = Image.open(img)
            to_tensor = transforms.ToTensor()
            actual_img = to_tensor(actual_img)
            means += torch.mean(actual_img, dim=(1, 2))
            stds += torch.std(actual_img, dim=(1, 2))

        means /= len(data)
        stds /= len(data)

        return means, stds
    
    def prepare_transform(self, means, stds, mode='train'):
        if mode == 'train':
            train_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=means,
                                    std=stds)
            ])
            self.transform = train_transforms
        elif mode == 'test' or mode == 'valid':
            test_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=means,
                                    std=stds)
            ])
            self.transform = test_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, class_id = self.data[idx]
        img = Image.open(image_name)
        # class_id = self.labels[idx]
        img_tensor = self.transform(img)
        class_id = torch.tensor([class_id])
        lab_tensor = torch.squeeze(class_id)
        return img_tensor, lab_tensor, image_name

