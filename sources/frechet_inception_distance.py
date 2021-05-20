import torch.nn as nn
import torchvision
import variable as var
import torch.nn.functional as F
import torch
import numpy as np
from scipy import linalg


class InceptionV3:
    def __init__(self):
        self.inception = torchvision.models.inception_v3(pretrained=True,
                                                         aux_logits=False).to(var.device)
        self.inception.fc = nn.Identity()
        self.inception.eval()

    def get_features_representation(self, x):
        with torch.no_grad():
            return self.inception(x)


def calculate_activation_statistics(images, inception_v3):
    """Calculates the statistics used by FID
    Args:
        images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
        inception_v3: inception v3 pretrained
    Returns:
        mu:     mean over all activations from the last pool layer of the inception model
        sigma:  covariance matrix over all activations from the last pool layer
                of the inception model.
    """
    feat_rep = inception_v3.get_features_representation(images)
    feat_rep = feat_rep.cpu().numpy()
    mu = np.mean(feat_rep, axis=0)
    sigma = np.cov(feat_rep, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            # m = np.max(np.abs(covmean.imag))
            # raise ValueError("Imaginary component {}".format(m))
            return np.nan
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid(real, fake, inception_v3, resize=None):
    """ Calculate FID between images1 and images2
    Args:
        real: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        fake: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        inception_v3: inception v3 pretrained
        resize: whether or not we want to resize our images (tuple of shape)
    Returns:
        FID (scalar)
    """
    if resize:
        real = F.interpolate(real, size=resize, mode='bicubic', align_corners=False)
        fake = F.interpolate(fake, size=resize, mode='bicubic', align_corners=False)

    mu_real, sigma1_real = calculate_activation_statistics(real, inception_v3)
    mu_fake, sigma_fake = calculate_activation_statistics(fake, inception_v3)
    fid = calculate_frechet_distance(mu_real, sigma1_real, mu_fake, sigma_fake)
    return fid