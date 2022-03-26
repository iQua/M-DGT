#!/usr/bin/env python
# -*- coding: utf-8 -*-



''' Inherent libs '''
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

''' Local libs '''
from datasets.provider_root import DatasetProviderROOT 



DataAnnos = namedtuple('annos', ['caption', 'caption_phrases', 'caption_phrase_bboxs', 
                                        'caption_phrases_cate', 'caption_phrases_cate_id'])

''' Detailed structure
    caption : a nested list, such as [['man', 'with', 'bat']], each item is a list that contains the phrases of the caption
    caption_phrases: the same structure as caption,  such as [['man', 'with', 'bat']]
    caption_phrases_cate: a nested list, each item is a string that presents the category of the phrase, such as [['person']]
    caption_phrases_cate_id: a list, each item is a int that shows the integar of the phrase, such as [1]
    caption_phrase_bboxs: a 2-depth nested list, each item is a nested list that presents the boxes of the phrase,
                            such as [[(85.125, 0.0, 186.55312500000002, 153.8550724637681)]].
'''


class ReferItGameProvider(DatasetProviderROOT):
    ''' Dataset Provider for ReferItGame
        
        Inputs:
            referitgame_base: an instance of the qmnist_base class

        Output:
            provide batch of images, labels = [digital, color_id, [ymin, xmin, ymax, xman]]
    '''

    def __init__(self, base_data, batch_size, epoch_size, num_data_loading_workers, 
                        transform_image_dec_func=None, transform_text_func=None, phase="train"):

        super().__init__(base_data, batch_size, epoch_size, num_data_loading_workers,
                        transform_image_dec_func,  transform_text_func, phase)      

        self.referitgame_base = self.base_data

        # obtain the corresponding elements in this phase
        self.mode_elements_holder, self.mode_flatten_emelemts = self.referitgame_base.get_phase_data(phase)


    def get_one_sample(self, sample_idx):
        [image_id, image_file_path, caption, caption_phrases, caption_phrase_bboxs, 
                caption_phrases_cate, caption_phrases_cate_id] = self.mode_flatten_emelemts[sample_idx]

        sample_name = image_id
        image_data = self.referitgame_base.get_image_data(image_id)

        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        
        ori_image_data = image_data.copy()


        caption = caption if any(isinstance(boxes_i, list) for boxes_i in caption) \
                                            else [caption]
        caption_phrase_bboxs = caption_phrase_bboxs if any(isinstance(boxes_i, list) for boxes_i in caption_phrase_bboxs) \
                                                    else [caption_phrase_bboxs]
        caption_phrases = caption_phrases if any(isinstance(boxes_i, list) for boxes_i in caption_phrases) \
                                            else [caption_phrases]
        caption_phrases_cate = caption_phrases_cate if any(isinstance(boxes_i, list) for boxes_i in caption_phrases_cate) \
                                            else [[caption_phrases_cate]]
        caption_phrases_cate_id = caption_phrases_cate_id if isinstance(caption_phrases_cate_id, list) \
                                            else [caption_phrases_cate_id]


        assert len(caption_phrase_bboxs) == len(caption_phrases)
        if self.transform_image_dec_func is not None:


            transformed = self.transform_image_dec_func(image=image_data, 
                                                        bboxes=caption_phrase_bboxs,
                                                        category_ids=caption_phrases_cate_id)
            

            image_data = transformed["image"]
            image_data = torch.from_numpy(image_data)
            caption_phrase_bboxs = transformed["bboxes"]

        if self.transform_text_func is not None:
            caption_phrase = self.transform_text_func(caption_phrase)
        
        caption_phrase_bboxs = [caption_phrase_bboxs] # convert to the standard structure

        sample_annos = DataAnnos(caption=caption, caption_phrases=caption_phrases, 
                                caption_phrase_bboxs=caption_phrase_bboxs, 
                                caption_phrases_cate=caption_phrases_cate, 
                                caption_phrases_cate_id=caption_phrases_cate_id)


        return sample_name, ori_image_data, image_data, sample_annos



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
            np.random.seed(int(round(delay * 1000))*rand_seed)
            one_sample = self.get_one_sample(iter_idx)

            return one_sample

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(len(self.mode_flatten_emelemts)), load=load_function)
        
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_data_loading_workers),
            shuffle=(False if self.is_eval_mode else True),
            collate_fn=collate_fn)

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

        
    def __len__(self):
        return int(self.epoch_size / self.batch_size)

