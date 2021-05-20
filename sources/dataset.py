import torch
import numpy as np
import os
import glob
from PIL import Image
import random
from torchvision import transforms
import pandas as pd
from copy import deepcopy
import variable as var
import utils as ut


class IconClassDataset(torch.utils.data.Dataset):
    def __init__(self, images, greyscale=False, sub_percent=None, resize=None):
        self.dir = os.path.join(os.environ["ICONCLASS"])
        self.greyscale = greyscale
        self.images = images

        if not resize:
            resize = [64, 64]

        self.general_transform = transforms.Compose([
                            transforms.Resize(resize),
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


class PokemonGen3Dataset(torch.utils.data.Dataset):
    def __init__(self, greyscale=False, n_item=None, resize=None, frame2=False, shiny=False, label=False):
        self.dir = os.path.join(os.environ["DATASETS"], 'pokemon_sprites/emerald/')
        self.dir_shiny = os.path.join(self.dir, 'shiny/')
        self.dir_frame2 = os.path.join(self.dir, 'frame2/')

        self.greyscale = greyscale

        self.images = ut.list_files(os.path.join(self.dir, "*"))
        self.images_shiny = ut.list_files(os.path.join(self.dir_shiny, "*"))
        self.images_frame2 = ut.list_files(os.path.join(self.dir_frame2, "*"))

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
            pokemon_numbers[ims_path] = int(ut.only_numerics(pokemon_number))

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


class PokemonGensDataset(torch.utils.data.Dataset):
    def __init__(self, label_name=None, greyscale=False, normalize=None,
                 n_item=None, resize=None, gens_to_remove=None):

        # global dir name, by generation dir name, images path
        self.dir = os.path.join(os.environ["DATASETS"], 'pokemon_sprites_processed/')
        self.gens_dir = self.get_gens_dir(gens_to_remove)
        self.images = self.get_images()
        self.greyscale = greyscale

        # load pokemon label
        self.path_to_label_id = None
        self.label_id_to_label_name = None
        self.path_to_pok_num = None
        if label_name:
            self.load_labels(label_name)

        # resize images parameters
        if not resize:
            resize = [64, 64]

        # image preprocessing
        self.general_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

        if normalize == "z_norm":
            self.normalize_transform = transforms.Normalize(var.NORM_MEAN, var.NORM_STD)
        elif normalize == "min_max_scale":
            self.normalize_transform = ut.MinMaxScaler1Neg1(var.NORM_MIN, var.NORM_MAX)
        elif normalize == 'tanh_estimator':
            self.normalize_transform = ut.TanhEstimator(var.NORM_MEAN, var.NORM_STD)
        else:
            self.normalize_transform = None

        # n_item dataset reduction (for unit test)
        if n_item:
            self.images = np.random.choice(self.images, size=(n_item, )).tolist() * (1000//n_item)
            self.update_labels_n_item()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label = 0
        if self.path_to_label_id:
            label = self.path_to_label_id[self.images[index]]

        image = Image.open(os.path.join(self.dir, self.images[index]))

        if self.greyscale:
            image = image.convert('L')
        else:
            image = image.convert('RGB')

        image = self.general_transform(image)
        if self.normalize_transform:
            image = self.normalize_transform(image)

        return image, label

    def get_gens_dir(self, gens_to_remove):

        all_gens_dir = ut.list_dirs(os.path.join(self.dir, '*'))
        if not gens_to_remove:
            return all_gens_dir

        gens_dir = list()
        for index, gen_dir in enumerate(all_gens_dir):
            if not any(gen_to_remove in gen_dir for gen_to_remove in gens_to_remove):
                gens_dir.append(gen_dir)

        return gens_dir

    def get_images(self):
        images = list()
        for gen_dir in self.gens_dir:
            images += ut.list_files(os.path.join(gen_dir, "*"))
        return images

    def load_labels(self, label_name):
        path_to_pok_num = dict()
        duplicates_counter = {duplicate_id: 0 for duplicate_id in var.duplicates_pk_num}
        n_max_duplicate = 8
        # extract pokemon number from file name & store it in path_to_pok_num dict
        for im_path in self.images:
            ims_paths_copy = deepcopy(im_path)
            for gen_dir in self.gens_dir:
                ims_paths_copy = ims_paths_copy.replace(gen_dir, '')

            pokemon_number = int(ut.only_numerics(ims_paths_copy.replace('.jpg', '')))

            if pokemon_number in var.duplicates_pk_num:
                duplicates_counter[pokemon_number] += 1
                if duplicates_counter[pokemon_number] >= n_max_duplicate:
                    continue

            path_to_pok_num[im_path] = pokemon_number

        # load pokemon attributes csv file to extract labels
        pokemon_attributes_file = os.path.join(self.dir, 'pokemon.csv')
        pok_attributes = pd.read_csv(pokemon_attributes_file)

        # filter with pokemon present in dataset
        pokemon_numbers_unique = np.unique(tuple(path_to_pok_num.values()))
        pok_attributes_filtered = pok_attributes.loc[pok_attributes.Number.isin(pokemon_numbers_unique),
                                                     [label_name, 'Number']]
        # extract unique labels & pokemon_number_to_labels
        unique_labels = pok_attributes_filtered.loc[:, label_name].unique()
        pok_num_to_label = pok_attributes_filtered.set_index('Number')[label_name].to_dict()

        # create labels_name_to_label_id mapping dict & pok_num_to_label_id mapping dict
        label_name_to_label_id = {label: index for index, label in enumerate(unique_labels)}
        pok_num_to_label_id = {_pok_num: label_name_to_label_id[_label] for _pok_num, _label in pok_num_to_label.items()}

        # remove unlabelled pokemon in path to pokemon id mapping dict
        self.path_to_pok_num = {path: pok_num for path, pok_num in path_to_pok_num.items()
                                if pok_num in pok_num_to_label_id.keys()}

        # create pokemon image path to label id mapping dict & label id to label name mapping dict
        self.path_to_label_id = {path: pok_num_to_label_id[pok_num]
                                 for path, pok_num in self.path_to_pok_num.items()}
        self.label_id_to_label_name = {label_id: label_name for label_name, label_id in label_name_to_label_id.items()}

        # remove image path from self.images that has no label
        filtered_images_path = [path for path, _ in self.path_to_pok_num.items()]
        self.images = filtered_images_path

    def update_labels_n_item(self):
        n_item_label_id_map = {self.path_to_label_id[im_path]: idx for idx, im_path in enumerate(np.unique(self.images))}
        self.path_to_label_id = {k: n_item_label_id_map[v] for k, v in self.path_to_label_id.items()
                                 if k in self.images}
        self.label_id_to_label_name = {n_item_label_id_map[k]: v for k, v in self.label_id_to_label_name.items()
                                       if k in n_item_label_id_map.keys()}

    def describe(self):
        return {
            'name': self.__class__.__name__,
            'gens_dir': '_'.join([gen_dir.replace(self.dir, '') for gen_dir in self.gens_dir]),
            'n_element': int(len(self.images))
        }
