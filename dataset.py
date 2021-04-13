import torch
import numpy as np
import os
import glob
from PIL import Image
import random
from torchvision import transforms
import pandas as pd


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


class PokemonDataset(torch.utils.data.Dataset):
    def __init__(self, greyscale=False, n_item=None, resize=None, frame2=False, shiny=False, label=False):
        self.dir = os.path.join(os.environ["DATASETS"], 'pokemon_sprites/emerald/')
        self.dir_shiny = os.path.join(self.dir, 'shiny/')
        self.dir_frame2 = os.path.join(self.dir, 'frame2/')

        self.greyscale = greyscale

        self.images = list_files(os.path.join(self.dir, "*"))
        self.images_shiny = list_files(os.path.join(self.dir_shiny, "*"))
        self.images_frame2 = list_files(os.path.join(self.dir_frame2, "*"))

        self.images += self.images_shiny if shiny else []
        self.images += self.images_frame2 if frame2 else []

        if not resize:
            resize = [64, 64]

        self.general_transform = transforms.Compose([
                            transforms.Resize(resize),
                            transforms.ToTensor()
        ])
        if n_item:
            self.images = np.random.choice(self.images, size=(n_item, )).tolist() * (1000//n_item)

        self.path_to_label = None
        self.label_map = None
        self.load_labels('type1', frame2, shiny)

    def load_labels(self, label_name, frame2, shiny):
        pokemon_numbers = dict()
        for ims_path in self.images:
            pokemon_number = ims_path.replace(self.dir, '').replace('.png', '')
            if frame2:
                pokemon_number = pokemon_number.replace('frame2/', '')
            if shiny:
                pokemon_number = pokemon_number.replace('shiny/', '')
            pokemon_numbers[ims_path] = int(only_numerics(pokemon_number))

        pokemon_attributes_file = os.path.join(os.environ["DATASETS"], 'pokemon_sprites/pokemon.csv')
        pok_attributes = pd.read_csv(pokemon_attributes_file)

        labels = pok_attributes.loc[pok_attributes.pokedex_number.isin(pokemon_numbers.values()), 'type1'].unique()
        pok_num_to_label = pok_attributes.set_index('pokedex_number')[label_name].to_dict()

        labels = {label: index for index, label in enumerate(labels)}
        self.path_to_label = {path: labels[pok_num_to_label[pok_num]] for path, pok_num in pokemon_numbers.items()}
        self.label_map = {index: label for index, label in enumerate(labels)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label = self.path_to_label[self.images[index]]
        image = Image.open(os.path.join(self.dir, self.images[index]))

        if self.greyscale:
            image = image.convert('L')
        else:
            image = image.convert('RGB')

        image = self.general_transform(image)
        return image, label


def list_files(path):
    return [log for log in glob.glob(path) if not os.path.isdir(log)]


def only_numerics(seq):
    seq_type= type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))