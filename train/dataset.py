from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from PIL import Image

from train import config


def get_data_loaders(dataset_path: Path) -> tuple[DataLoader, DataLoader]:
    """Returns train and test data loaders"""
    ds = ImageFolder(str(dataset_path))
    train_data, test_data, _, _ = train_test_split(
        ds.imgs,
        ds.targets,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    train_transform = get_test_train_transform()
    test_transform = get_test_train_transform()

    train_loader = DataLoader(
        dataset=ImageLoader(train_data, train_transform),
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=ImageLoader(test_data, test_transform),
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    return train_loader, test_loader


def get_test_train_transform():
    """Return transformations both for train and test"""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3)
    ])


def get_inference_transform():
    """Returns scriptable inference transform"""
    return nn.Sequential(
        T.Resize((224, 224)),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.5]*3, [0.5]*3)
    )


class ImageLoader(Dataset):
    def __init__(
        self,
        dataset: list[tuple[str, int]],
        transform: Optional[T.Compose] = None
    ):
        self._dataset = self._prepare_dataset(dataset)
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        image = Image.open(self._dataset[item][0])
        class_idx = self._dataset[item][1]
        if self._transform:
            image = self._transform(image)
        return image, class_idx

    def _prepare_dataset(
        self,
        dataset: list[tuple[str, int]]
    ) -> list[tuple[str, int]]:
        """Filters ds and keeps only RGB images"""
        return [
            (img_path, class_idx)
            for img_path, class_idx in dataset
            if Image.open(img_path).getbands() == ('R', 'G', 'B')
        ]
