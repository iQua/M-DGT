#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Inherent Python '''
import os
from collections import namedtuple, OrderedDict
import functools
import re
import json
import time
''' Third Libs '''
import numpy as np
import torch
import torchnet as tnt
from skimage import io, color
from sklearn.utils import shuffle
import cv2
''' Local Libs '''
from datasets.provider_root import DatasetProviderROOT
''' Detailed structure
    digitalNumbers : a nested list, such as [[6], [4], [3], [8], [9]], each item is a list that contains the phrases of the caption
    digitalColors: the same structure as caption,  such as [['man', 'with', 'bat']]
    digitalBoxes: a 2-depth nested list, each item is a nested list that presents the boxes of the phrase,
                            such as [[(85.125, 0.0, 186.55312500000002, 153.8550724637681)]].
'''


def is_pattern_file_exist(file_dir, pattern):
    flag = False
    files = os.listdir(file_dir)
    pattern = re.compile(pattern)

    for file in files:
        m = pattern.match(file)
        if m is not None:
            flag = True
    return flag


class QMNISTProvider(DatasetProviderROOT):
    ''' Dataset Provider for Flicker30K
        
        Inputs:
            base_data: an instance of the qmnist_base class

        Output:
            provide batch of images, labels = [digital, color_id, [ymin, xmin, ymax, xman]]
    '''
    def __init__(self,
                 base_data,
                 batch_size,
                 epoch_size,
                 num_data_loading_workers,
                 transform_image_dec_func=None,
                 transform_text_func=None,
                 phase="train"):

        super().__init__(base_data, batch_size, epoch_size,
                         num_data_loading_workers, transform_image_dec_func,
                         transform_text_func, phase)

        self.qmnist_base = self.base_data

        self.phase_data_info = self.qmnist_base.get_phase_data(phase)
        self.phase_samples_name = list(self.phase_data_info.keys())
        self.phase_samples_count = len(self.phase_data_info.keys())

    def get_one_sample(self, sample_idx):
        """ Sample a one sample from the dataset of the underlying phase

        Returns:
            [episode_exampler] (a class): [an instance of the class MetaTaskExampler]
        """
        sample_name = self.phase_samples_name[sample_idx]
        phase_sample_annos = self.phase_data_info[
            sample_name]  # get the annotations of this sample

        digital_nums, digital_colors, sample_bboxes, phrases, caption = self.qmnist_base.decode_annos(
            phase_sample_annos)

        sample_bboxes = self.qmnist_base.convert_boxes(sample_bboxes)

        # get the sample data - the image data
        sample_img_path = os.path.join(self.qmnist_base.images_dir,
                                       sample_name + ".png")

        image_data = cv2.imread(sample_img_path)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        ori_image_data = image_data.copy()

        if self.transform_image_dec_func is not None:
            transformed = self.transform_image_dec_func(
                image=image_data,
                bboxes=sample_bboxes,
                category_ids=digital_nums)

            image_data = transformed["image"]
            image_data = torch.from_numpy(image_data)
            sample_bboxes = transformed["bboxes"]
            digital_nums = transformed["category_ids"]

        if self.transform_text_func is not None:
            digital_nums, phase_sample_text, digital_bboxs, phrases, caption = self.qmnist_base.decode_annos(
                phase_sample_annos)
            phrases = self.transform_text_func(phrases)

        # conver the digital number/color to the standard
        digital_nums = [[di_n] for di_n in digital_nums]
        digital_colors = [[di_c] for di_c in digital_colors]
        sample_bboxes = [[di_b] for di_b in sample_bboxes]
        phrases = [[ph] for ph in phrases]
        caption = [caption]

        phase_sample_annos = self.qmnist_base.set_numbers(
            phase_sample_annos, digital_nums)
        phase_sample_annos = self.qmnist_base.set_colors(
            phase_sample_annos, digital_colors)

        phase_sample_annos = self.qmnist_base.set_boxes(
            phase_sample_annos, sample_bboxes)

        phase_sample_annos = self.qmnist_base.set_phrases_caption(
            phase_sample_annos, phrases, caption)

        return sample_name, ori_image_data, image_data, phase_sample_annos

    def get_iterator(self, epoch=0):
        if epoch == 0:
            epoch = epoch + 1
        rand_seed = epoch
        start_time = time.time()

        # The random cannot be placed here! This will make the same sampled for batches under this epoch
        # random.seed(rand_seed)
        # np.random.seed(rand_seed)
        def collate_fn(batch):
            """[The construction of the loaded batch of data]

            Args:
                batch ([list]): [a list in which each element contains the data for one task,
                                assert len(batch) == number of tasks,
                                assert len(batch[i]) == 6 that is the output of create_task_examples_data function]

            Returns:
                [batch]: [return the original batch of data directly]
            """
            return batch

        def load_function(iter_idx):
            current_time = time.time()
            delay = current_time - start_time
            np.random.seed(int(round(delay * 1000)) * rand_seed)
            one_sample = self.get_one_sample(iter_idx)

            return one_sample

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(
            self.epoch_batches_size),
                                              load=load_function)

        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else
                         self.num_data_loading_workers),
            shuffle=(False if self.is_eval_mode else True),
            collate_fn=collate_fn)

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)
