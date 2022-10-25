import torch

def f(t, T, s=0.008):
    return torch.cos(((t / T + s) / (1 + s)) * (torch.pi / 2)) ** 2

def alphabar_t(t, T, s):
    return f(t, T, s) / f(torch.tensor(0), T, s)

def alpha_t(t, abar):
    return torch.clip(abar[t] / abar[t - 1], 0.001, 0.999)


def get_cached(T, s=0.008):
    return alphabar_t(torch.arange(0, T+1),T, s)
