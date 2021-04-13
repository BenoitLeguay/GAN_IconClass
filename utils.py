import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision.utils import make_grid


def rolling_window(a, window=3):
    if type(a) == list:
        a = np.array(a)

    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window - 1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad, mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def show_tensor_images(image_tensor, num_images=10):
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def return_tensor_images(image_tensor, num_images=10):
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    return image_grid


def critic_layer(n_features):
    return nn.Sequential(
        nn.Conv2d(n_features, n_features * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(n_features * 2),
        nn.LeakyReLU(0.2, inplace=True)
    )


def generator_layer(n_features):
    return nn.Sequential(
        nn.ConvTranspose2d(n_features * 2, n_features, 4, 2, 1, bias=False),
        nn.BatchNorm2d(n_features),
        nn.Dropout(0.5),
        nn.LeakyReLU(0.2, inplace=True),
    )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)