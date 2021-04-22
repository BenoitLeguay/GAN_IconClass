import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NORM_MEAN = [0.9115, 0.8998, 0.8904]
NORM_STD = [0.1860, 0.1991, 0.2153]
NORM_MAX = [1.0, 1.0, 1.0]
NORM_MIN = [0.0, 0.0, 0.0]

duplicates_pk_num = [676, 671, 670, 669, 666, 493, 479, 201, 25]
body_type_groups_7 = {'four_wings': 'wings',
                      'two_wings': 'wings',
                      'head_base': 'head',
                      'head_only': 'head',
                      'head_legs': 'head',
                      'head_arms': 'head',
                      'bipedal_tailed': 'bipedal_tailed',
                      'bipedal_tailless': 'bipedal_tailless',
                      'quadruped': 'quadruped',
                      'serpentine_body': 'serpentine',
                      'with_fins': 'serpentine',
                      'insectoid': 'insectoid_multiple_bodies',
                      'multiple_bodies': 'insectoid_multiple_bodies',
                      'several_limbs': 'insectoid_multiple_bodies'}

body_type_groups_10 = {'four_wings': 'wings',
                       'two_wings': 'wings',
                       'head_base': 'head_base',
                       'head_only': 'head_only',
                       'head_legs': 'bipedal_tailless',
                       'head_arms': 'multiple_bodies',
                       'bipedal_tailed': 'bipedal_tailed',
                       'bipedal_tailless': 'bipedal_tailless',
                       'quadruped': 'quadruped',
                       'serpentine_body': 'serpentine',
                       'with_fins': 'with_fins',
                       'insectoid': 'insectoid_several_limbs',
                       'multiple_bodies': 'multiple_bodies',
                       'several_limbs': 'insectoid_several_limbs'}