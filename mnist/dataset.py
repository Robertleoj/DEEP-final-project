
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trans
import random
import torch
import math

from torchvision.datasets import ImageFolder
from utils import get_cached


class DiffDset(Dataset):

    def __init__(self, dset, T, s=0.008, rotate=0):
        self.dset = dset
        self.T = T
        self.abar = get_cached(T)
        self.rotate = rotate
        
    def __getitem__(self, idx, t=None):
        img: torch.Tensor = self.dset[idx][0]

        if self.rotate:
            img = trans.functional.rotate(img, -90)

        if t is None:
            t = random.randint(1, self.T)

        eps = torch.normal(0, 1, size=img.size())


        noise_img = torch.sqrt(self.abar[t]) * img + math.sqrt(1 - self.abar[t]) * eps

        return noise_img, eps, img, t

    def __len__(self):
        return len(self.dset)

def get_pokemon(T, image_size, path):

    transf = trans.Compose([
        trans.Resize((image_size, image_size)),
        trans.ToTensor(),
        trans.RandomHorizontalFlip()
    ])

    dset1 = ImageFolder(path, transform=transf)

    dset_og = DiffDset(dset1, T)

    return dset_og

def get_dataset(T, image_size, split='balanced'):

    transf = trans.Compose([
        trans.Resize((image_size, image_size)),
        trans.ToTensor(),
        trans.RandomVerticalFlip(p = 1)
    ])

    data = datasets.EMNIST('./data', split=split, transform=transf, download=True)
    train_set = DiffDset(data, T)
    return train_set

