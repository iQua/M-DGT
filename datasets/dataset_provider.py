#!/usr/bin/env python
# -*- coding: utf-8 -*-



''' Inherent libs '''
import os
import re
import time 

''' Third libs '''
import torchnet as tnt
import numpy as np
import torch
import pandas as pd
from skimage import io, color
from sklearn.utils import shuffle
from PIL import Image


''' Local libs '''



class DataSetProvider(object):
    
    def __init__(self, base_dataset_root, transform_image_func=None, transform_text_func=None, phase="trian"):
        
        assert phase in ["train", "test", "val"]

        self.base_dataset_root = base_dataset_root
        self.phase = phase 

        self.transform_image_func = transform_image_func
        self.transform_text_func = transform_text_func

        self.phase_dataset_images_dir = os.path.join(self.meta_dataset_root.get_save_splited_data_dir(), phase)
    
        # Count the total number of images and labels 
        self._phase_category_names = [cate_name for cate_name in os.listdir(self.phase_dataset_images_dir) if not cate_name.startswith('.')]
        self._phase_category_names = sorted(self._phase_category_names)

        self._number_of_category = len(self._phase_category_names)
        self._number_of_images = sum([len(files) for r, d, files in os.walk(self.phase_dataset_images_dir)])

        sorted_phase_category_names = sorted(self._phase_category_names)

        self._category_id_mapper = dict(zip(sorted_phase_category_names, list(range(self._number_of_category))))
        self._id_category_mapper = dict(zip(list(range(self._number_of_category)), sorted_phase_category_names))


    