import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CarsDataSet(Dataset):
    def __init__(self, path):
        self.data_path = path

        image_folder = os.listdir(self.data_path)
        self.train_dataset = [self.data_path + image for image in image_folder ]

        self.original_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.lr_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.
            transforms.ToTensor()
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


def get_dataset(train_directory, test_directory):
    train_set = CarsDataSet(train_directory)
    test_set = CarsDataSet(test_directory)
    return train_set, test_set


def make_dataloader(train_set, test_set, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_dataloader(train_directory, test_directory, batch_size):
    return make_dataloader(*get_dataset(train_directory, test_directory), batch_size=batch_size)
