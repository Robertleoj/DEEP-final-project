
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trans
import random
import torch
import math

from utils import get_cached


class DiffDset(Dataset):

    def __init__(self, dset, T, s=0.008):
        self.dset = dset
        self.T = T
        self.abar = get_cached(T)
        
    def __getitem__(self, idx, t=None):
        img: torch.Tensor = self.dset[idx][0]
        if t is None:
            t = random.randint(1, self.T)

        eps = torch.normal(0, 1, size=img.size())


        noise_img = torch.sqrt(self.abar[t]) * img + math.sqrt(1 - self.abar[t]) * eps


        return noise_img, eps, img, t

    def __len__(self):
        return len(self.dset)

def get_dataset(T):

    transf = trans.Compose([
        trans.ToTensor()
    ])

    data = datasets.CIFAR10('./data', train=True, transform=transf, download=True)
    train_set = DiffDset(data, T)
    return train_set

