#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Inherent libs '''
import os
''' Third libs '''
import numpy as np
from absl import flags
''' Local libs '''

FLAGS = flags.FLAGS

CURRENT_PROJECT_DIR = os.path.abspath(os.getcwd())

flags.DEFINE_boolean("enable_CUDA", True,
                     "Whether to use the cuda to train the model")
flags.DEFINE_string("gpu_devices", "0, 1, 2, 3",
                    "The GPU devices used for learning")

SOURCE_data_path = "source_datasets/CV_datasets"

#########################################################
############ Dataset hyperparameters ####################
#########################################################
flags.DEFINE_list("Supported_Datasets_Names",
                  ['QMNIST', 'Flickr30KEntities', 'ReferItGame'],
                  "The name of datasets that are supoorted in our work")

################## Source Datasets ######
flags.DEFINE_string("QMNIST_source_path",
                    os.path.join(SOURCE_data_path, "MNIST_Related/QMNIST"),
                    "The source data path of the Cifar100 dataset")

flags.DEFINE_string("Flickr30KEntities_source_path",
                    os.path.join(SOURCE_data_path, "Flickr30k-entities"),
                    "The source data path of the ImageNet dataset")

flags.DEFINE_string("ReferItGame_source_path",
                    os.path.join(SOURCE_data_path, "ReferItGame"),
                    "The source data path of the ReferItGame dataset")

flags.DEFINE_string("COCO_source_path",
                    os.path.join(SOURCE_data_path, "COCO/COCO-2017"),
                    "The source data path of the COCO dataset")

flags.DEFINE_string(
    "COCO_source_images_path",
    os.path.join(SOURCE_data_path, "COCO/COCO-2017/source_images"),
    "The source data path of the images of the COCO dataset")

################## Created Datasets ######
flags.DEFINE_string("QMNIST_path",
                    os.path.join(CURRENT_PROJECT_DIR, "datasets", "QMNIST"),
                    "The data path of the QMNIST dataset")

flags.DEFINE_string(
    "F30kE_path",
    os.path.join(CURRENT_PROJECT_DIR, "datasets", "Flickr30kEntities"),
    "The data path of the Flickr30kEntities dataset")

flags.DEFINE_string(
    "RIG_path", os.path.join(CURRENT_PROJECT_DIR, "datasets", "ReferItGame"),
    "The data path of the ReferItGame dataset")

flags.DEFINE_integer(
    "num_data_loading_workers", 0,
    "the number of workers used for data loading (default 0 to use the main proces)"
)
