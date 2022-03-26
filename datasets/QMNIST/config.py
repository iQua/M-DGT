#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Inherent Python '''
import os


''' Third Libs '''


''' Local Libs '''


'''
        'items_to_descriptions': {
            'images': 'images',
            'img_number': 'id of images',
            'digital': 'the digital needed to be searched',
            '[ymin, xmin, ymax, xman]': 'list of integer ids corresponding to the caption words'
        }
'''

colors_digial_map = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'white': 4, 'saddlebrown': 5, 'brown': 6, 'aqua': 7}
digital_colors_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'white', 5: 'saddlebrown', 6: 'brown', 7: 'aqua'}
colors_digial_map = colors_digial_map

ROOT_DATA_DIR = os.path.join("datasets", "CV_datasets", "MNIST_Related", "QMNIST")

QMNIST_CONFIGS = {
    'name': 'qmnist',
    # root directory of dataset
    'root_ds': ROOT_DATA_DIR,
    #'vocabulary_map_file_name': 'vocabulary_map.pkl',
    'img_suffix': '.jpg',
    'img_format':  b'JPEG',
    'num_classes': None,
    "box_coordinates_type": "ymin, xmin, ymax, xman",
    'features': ["images",
                    "img_number,",
                    "digital",
                    "[ymin, xmin, ymax, xman]"],

    "Default_Splits_Config": {
            'train': {
                'size': 25000,
                'pattern': r'train.*\.npy'
            },
            'test': {
                'size': 64008,
                'pattern': r'test.*\.npy'
            },
            'val': {
                'size': 1000,
                'pattern': None
            },
            'inference': {
                'size': None,
                'pattern': 'None'
            }
    }
}