import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NORM_MEAN = [0.9115, 0.8998, 0.8904]
NORM_STD = [0.0667, 0.0663, 0.0774]