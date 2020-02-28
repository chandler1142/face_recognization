import numpy as np
import torch
from torch.autograd import Variable


def flattern(path):
    vector = np.load(path)
    result = vector.flatten()
    return result


def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
