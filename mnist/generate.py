# Cell
import torch
from utils import get_cached, alpha_t
import math
from torchvision.utils import save_image
import os
from tqdm import tqdm


def mu_t(t, x_t, pred, cached):
    a_t = alpha_t(t, cached)

    return (
        (1 / torch.sqrt(a_t)) * x_t
        - (
            (1 - a_t) 
            / (torch.sqrt(1 - cached[t]) * torch.sqrt(a_t))
        ) * pred
    )
    # return (

    #     torch.sqrt(a_t).cuda() * (1 - cached[t - 1]) * x_t
    #     + torch.sqrt(cached[t - 1]).cuda() * (1 - a_t) * pred
    # ) / (1 - cached[t])

# Cell
def generate(n, T, net, shape=(1, 28, 28)):
    cached = get_cached(T).cuda()

    batch_shape = (n, *shape)

    images = torch.normal(0, 1, size=batch_shape).cuda()

    t_iter = T - 1

    for t_iter in tqdm(range(T - 1, 0, -1)):
        pred = net(images, torch.ones(n).cuda() * t_iter)
        images = mu_t(
            t_iter, 
            images, 
            pred, 
            cached
        )

    return images

def train_sample(n, T, net, epoch, folder='./samples', shape=(1, 28, 28)):

    fldr = f'{folder}/{epoch}'
    if not os.path.exists(fldr):
        os.mkdir(fldr)
    imgs = generate(n, T, net, shape)

    for i in range(n):
        save_image(imgs[i], f"{fldr}/{i}.png")


    