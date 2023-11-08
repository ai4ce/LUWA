import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import cv2
import numpy as np
import pandas as pd

csv_path = "/mnt/c/0_DATASET/AI_for_Tribology/LUA_Dataset_Nov/CSV/256_50x_6w_test.csv"
img_path = "/mnt/c/0_DATASET/AI_for_Tribology/LUA_Dataset_Nov/256/50x/texture"

def StatisticsImage(train_data):
    means = torch.zeros(3)
    stds = torch.zeros(3)

    for img, label in train_data:
        means += torch.mean(img, dim=(1, 2))
        stds += torch.std(img, dim=(1, 2))

    means /= len(train_data)
    stds /= len(train_data)

    print(f'Calculated means: {means}')
    print(f'Calculated stds: {stds}')
    return means, stds

def TrainTransform(train_data):
    means, stds = StatisticsImage(train_data)
    train_transforms = transforms.Compose([
        # transforms.Resize(pretrained_size),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomCrop(pretrained_size, padding=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=means,
                             std=stds)
    ])
def TestTransform(test_data):
    means, stds = StatisticsImage(test_data)
    test_transforms = transforms.Compose([
        # transforms.Resize(pretrained_size),
        # transforms.CenterCrop(pretrained_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=means,
                             std=stds)
    ])




class TribologyDataset(Dataset):
    """Tribology dataset."""

    def __init__(self, csv_path = csv_path, img_path = img_path, transform=transforms.ToTensor()):
        self.img_path = img_path
        self.csv_path = csv_path
        self.used_labels = ["antler", "beechwood", "beforeuse", "bone", "ivory",
                            "sprucewood"]

        self.labels_maps = {"antler": 0, "beechwood": 1, "beforeuse": 2, "bone": 3,
                            "ivory": 4, "sprucewood": 5, "barley": 6, "fern": 7, "horsetail": 8}

        labels_set = []

        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        # self.image_name = []
        # self.labels = []
        self.data = []

        for name, label in zip(self.image_name_all, self.labels_all):

            if label in self.used_labels:
                # self.labels.append(self.labels_maps[label])
                img_name = os.path.join(img_path, label, name)
                # self.image_name.append(img_name)
                self.data.append([img_name, self.labels_maps[label]])

        self.data_len = len(self.data)

        # self.image_name = np.asarray(self.image_name)
        # self.labels = np.asarray(self.labels)




        # self.imgs_path = os.path.join(root_dir, split) + '/'
        # file_list = glob.glob(self.imgs_path + "*.bmp")
        # self.data = []
        # for img_path in file_list:
        #     img_name = img_path.split("/")[-1]
        #     class_name = img_name.split("_")[3]
        #     self.data.append([img_path, class_name])
        # self.class_map = {'ANTLER': 0, 'BEECHWOOD': 1, 'BONE': 2, 'IVORY': 3, 'SPRUCEWOOD': 4, 'SPURCEWOOD': 4}
        # self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, class_id = self.data[idx]
        img = cv2.imread(image_name)
        # class_id = self.labels[idx]
        img_tensor = self.transform(img)
        class_id = torch.tensor([class_id])
        lab_tensor = torch.squeeze(class_id)
        return img_tensor, lab_tensor

