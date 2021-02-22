import matplotlib.pyplot as plt
import numpy as np
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
