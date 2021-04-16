import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision.utils import make_grid
import collections
import variable as var


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
    image_tensor = inverse_normalize(image_tensor)
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def return_tensor_images(image_tensor, num_images=10):
    image_tensor = inverse_normalize(image_tensor)
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


def flatten_dict(d, parent_key='', sep='_'):

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def inverse_normalize(batch_normalize):
    batch_inv_normalize = batch_normalize.new(*batch_normalize.size())
    batch_inv_normalize[:, 0, :, :] = batch_normalize[:, 0, :, :] * var.NORM_STD[0] + var.NORM_MEAN[0]
    batch_inv_normalize[:, 1, :, :] = batch_normalize[:, 1, :, :] * var.NORM_STD[1] + var.NORM_MEAN[1]
    batch_inv_normalize[:, 2, :, :] = batch_normalize[:, 2, :, :] * var.NORM_STD[2] + var.NORM_MEAN[2]

    return batch_inv_normalize


def compute_dataset_mean_std(dloader):
    mean, std = None, None
    for i_batch, (real, real_classes) in enumerate(dloader):
        mean = real.view(real.size(0), real.size(1), -1).mean(axis=2).mean(0)
        std = real.view(real.size(0), real.size(1), -1).std(axis=2).std(0)
    return mean, std
