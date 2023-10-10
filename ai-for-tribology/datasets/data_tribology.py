import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import cv2

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

    def __init__(self, root_dir, split='train', transform=transforms.ToTensor()):
        self.imgs_path = os.path.join(root_dir, split) + '/'
        file_list = glob.glob(self.imgs_path + "*.bmp")
        self.data = []
        for img_path in file_list:
            img_name = img_path.split("/")[-1]
            class_name = img_name.split("_")[3]
            self.data.append([img_path, class_name])
        self.class_map = {'ANTLER': 0, 'BEECHWOOD': 1, 'BONE': 2, 'IVORY': 3, 'SPRUCEWOOD': 4, 'SPURCEWOOD': 4, 'BU': 5}
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        class_id = self.class_map[class_name]
        img_tensor = self.transform(img)
        class_id = torch.tensor([class_id])
        lab_tensor = torch.squeeze(class_id)
        return img_tensor, lab_tensor

