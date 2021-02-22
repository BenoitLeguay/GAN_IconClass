import torch
import os
import glob
from PIL import Image
import random
from torchvision import transforms


class IconClassDataset(torch.utils.data.Dataset):
    def __init__(self, images, greyscale=False, sub_percent=None, resize=None):
        self.dir = os.path.join(os.environ["ICONCLASS"])
        self.greyscale = greyscale
        self.images = images

        if not resize:
            resize = [64, 64]

        self.general_transform = transforms.Compose([
                            transforms.Resize(resize),
                            transforms.ToTensor()#,
                            #transforms.Normalize([0.5], [0.5])
        ])
        if sub_percent:
            self.images = self.images_sub_percent(sub_percent)

    def images_sub_percent(self, sub_percent):
        random.shuffle(self.images)
        n_elements = len(self)
        trunc_low, trunc_high = int(n_elements*sub_percent[0]), int(n_elements*sub_percent[1])
        return self.images[trunc_low: trunc_high]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.dir, self.images[index]))
        if self.greyscale:
            image = image.convert('L')
        else:
            image = image.convert('RGB')

        image = self.general_transform(image)
        return image, 0


class CatsDatasets(torch.utils.data.Dataset):
    def __init__(self, greyscale=False, sub_percent=None):
        self.dir = os.path.join(os.environ["CATS_DOGS"])
        self.cat_dir = os.path.join(self.dir, "Cat")
        self.images = glob.glob(os.path.join(self.cat_dir, "*"))
        self.greyscale = greyscale
        self.general_transform = transforms.Compose([
                            transforms.Resize([64, 64]),
                            transforms.ToTensor()
        ])
        if sub_percent:
            self.images = self.images_sub_percent(sub_percent)

    def images_sub_percent(self, sub_percent):
        random.shuffle(self.images)
        n_elements = len(self)
        trunc_low, trunc_high = int(n_elements*sub_percent[0]), int(n_elements*sub_percent[1])
        return self.images[trunc_low: trunc_high]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if self.greyscale:
            image = image.convert('L')
        else:
            image = image.convert('RGB')

        image = self.general_transform(image)

        return image, 0

