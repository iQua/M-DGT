#!/usr/bin/env python
# -*- coding: utf-8 -*-



''' Inherent libs '''
import os
from abc import ABC, abstractstaticmethod


''' Third libs '''
import numpy as np



''' Local libs '''


class DatasetProviderROOT(ABC):
    """[The provider class for all datasets - I.e. The common rules required to be obey by all datasets]

        ABC ([type]): [Abstract Base Classes, 
                        This module provides the infrastructure for defining abstract base classes (ABCs) in Python]
    """
    def __init__(self, base_data, batch_size, epoch_size, num_data_loading_workers, 
                 transform_image_dec_func=None, transform_text_func=None, phase="train"):

        self.base_data = base_data

        self.phase = phase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.epoch_batches_size = batch_size * epoch_size # total_number_of_tasks_per_epoch
        self.num_data_loading_workers = num_data_loading_workers
        self.is_eval_mode = (self.phase=='test') or (self.phase=='val')

        self.transform_image_dec_func = transform_image_dec_func
        self.transform_text_func = transform_text_func



    @abstractstaticmethod
    def get_one_sample(self, sample_idx):
        """[Getting one sample from the dataset]

        Args:
            destination_dataset_dir ([string], optional): [The path of the directory that contains the splited dataset]. Defaults to None.
        """
        pass

    @abstractstaticmethod
    def get_iterator(self, epoch=0):
        pass

