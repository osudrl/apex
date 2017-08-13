import torch

def center(x):
    mean = x.mean().unsqueeze(0).expand_as(x)
    std = x.std().unsqueeze(0).expand_as(x)

    return (x - mean) / (std + 1e-8)
