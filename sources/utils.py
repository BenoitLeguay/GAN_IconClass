import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import collections
import glob
import variable as var
import torchvision.utils as vutils
import torch


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


def show_images_grid(batch, n_images=10):
    plt.imshow(images_grid(batch, n_images=n_images, tensorboard=False))
    plt.show()


def images_grid(batch, n_images=10, tensorboard=True):
    batch = batch.detach().cpu()
    grid = vutils.make_grid(
        batch[:n_images],
        nrow=5,
        padding=1,
        scale_each=False,
        normalize=True)
    if tensorboard:
        return grid
    return grid.permute(1, 2, 0).squeeze()


def return_tensor_images(image_tensor, num_images=10):
    print("deprecated function pls use images_grid")
    image_unflat = image_tensor.detach().cpu()
    image_grid = vutils.make_grid(image_unflat[:num_images], nrow=5)
    return image_grid


def critic_layer_test(n_features):
    return nn.Sequential(
        nn.Conv2d(n_features, n_features * 2, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(n_features * 2),
    )


def generator_layer_test(n_features):
    return nn.Sequential(
        nn.ConvTranspose2d(n_features * 2, n_features, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.5),
        nn.BatchNorm2d(n_features),
    )


def discriminator_layer(n_features):
    return nn.Sequential(
        nn.Conv2d(n_features, n_features * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(n_features * 2),
        nn.LeakyReLU(0.2, inplace=True)
    )


def critic_layer(n_features):
    return nn.Sequential(
        nn.Conv2d(n_features, n_features * 2, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True)
    )


def generator_layer(n_features):
    return nn.Sequential(
        nn.ConvTranspose2d(n_features * 2, n_features, 4, 2, 1, bias=False),
        nn.BatchNorm2d(n_features),
        nn.Dropout(0.5),
        nn.LeakyReLU(0.2, inplace=True),
    )


def generator_layer_up_sample(n_features):
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(n_features * 2, n_features, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(n_features),
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


def compute_dataset_mean_std(data_loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


def list_files(path):
    return [log for log in glob.glob(path) if not os.path.isdir(log)]


def list_dirs(path):
    return [log + "/" for log in glob.glob(path) if os.path.isdir(log)]


def only_numerics(seq):
    seq_type = type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))


class MinMaxScaler1Neg1(object):

    def __init__(self, data_min, data_max):
        self.min = data_min
        self.max = data_max

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        scaled_tensor = tensor.new(*tensor.size())
        scaled_tensor[0, :, :] = 2 * ((tensor[0, :, :] - self.min[0]) / (self.max[0] - self.min[0])) - 1
        scaled_tensor[1, :, :] = 2 * ((tensor[1, :, :] - self.min[1]) / (self.max[1] - self.min[1])) - 1
        scaled_tensor[2, :, :] = 2 * ((tensor[2, :, :] - self.min[2]) / (self.max[2] - self.min[2])) - 1

        return scaled_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(max={0}, min={1})'.format(self.max, self.min)


class TanhEstimator(object):

    def __init__(self, data_min, data_max):
        self.mean = data_min
        self.std = data_max

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        scaled_tensor = tensor.new(*tensor.size())
        scaled_tensor[0, :, :] = (1/2) * (torch.tanh(1e-2 * (tensor[0, :, :] - self.mean[0]) / self.std[0]) + 1)
        scaled_tensor[1, :, :] = (1/2) * (torch.tanh(1e-2 * (tensor[1, :, :] - self.mean[1]) / self.std[1]) + 1)
        scaled_tensor[2, :, :] = (1/2) * (torch.tanh(1e-2 * (tensor[2, :, :] - self.mean[2]) / self.std[2]) + 1)

        return scaled_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(max={0}, min={1})'.format(self.max, self.min)
