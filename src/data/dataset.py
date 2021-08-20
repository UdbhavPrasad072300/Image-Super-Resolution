import os

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CarsDataSet(Dataset):
    def __init__(self, path):
        self.data_path = path

        image_folder = os.listdir(self.data_path)
        self.train_dataset = [self.data_path + image for image in image_folder]

        self.original_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.], std=[1.]),
        ])

        self.lr_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[-1.], std=[2.]),
        ])

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        data = self.train_dataset[idx]
        image = Image.open(data)
        original_image = self.original_transform(image)
        sr_image = self.lr_transform(image)
        if original_image.size(0) == 1:
            original_image = original_image.repeat(3, 1, 1)
            sr_image = sr_image.repeat(3, 1, 1)

        return original_image, sr_image


# Source: https://www.kaggle.com/akhileshdkapse/super-image-resolution
class ImgSuperResolutionDataset(Dataset):
    def __init__(self, path="./data/raw/ImgSuperResolutionDataset"):
        super(ImgSuperResolutionDataset, self).__init__()

        self.hr_path = path + "/HR/"
        self.lr_path = path + "/LR/"

        hr_folder = os.listdir(self.hr_path)
        self.train_dataset = [image for image in hr_folder]

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.lr_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        data = self.train_dataset[idx]
        hr_image = Image.open(self.hr_path + data)
        lr_image = Image.open(self.lr_path + data)
        original_image = self.img_transform(hr_image)
        sr_image = self.lr_transform(lr_image)

        return original_image, sr_image

    def __len__(self):
        return len(self.train_dataset)


def get_cars_dataset(train_directory, test_directory):
    train_set = CarsDataSet(train_directory)
    test_set = CarsDataSet(test_directory)
    return train_set, test_set


def make_cars_dataloader(train_set, test_set, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_cars_dataloader(train_directory, test_directory, batch_size):
    return make_cars_dataloader(*get_cars_dataset(train_directory, test_directory), batch_size=batch_size)


def get_isr_dataset():
    train_set = ImgSuperResolutionDataset()
    train_set, test_set = torch.utils.data.random_split(train_set, [90, 10])
    return train_set, test_set


def make_isr_dataloader(train_set, test_set, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_isr_dataloader(batch_size):
    return make_isr_dataloader(*get_isr_dataset(), batch_size=batch_size)
