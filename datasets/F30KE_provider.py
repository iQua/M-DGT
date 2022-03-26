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
from .Flickr30kEntities import utils


DataAnnos = namedtuple('annos', ['caption', 'caption_phrases', 'caption_phrase_bboxs', 
                                        'caption_phrases_cate', 'caption_phrases_cate_id'])


''' Detailed structure
    caption : a nested list, such as [['The woman is applying mascara while looking in the mirror .']], 
    caption_phrases: a nested list, each item is a list that contains the phrases of the caption
                    such as [['Military personnel'], ['greenish gray uniforms'], ['matching hats']]
    caption_phrases_cate: a nested list, each item is a string that presents the categories of the phrase, 
                    such as [['people'], ['bodyparts'], ['other']]
    caption_phrases_cate_id: a list, each item is a int that shows the integar/str of the phrase, such as ['121973', '121976', '121975']
    caption_phrase_bboxs: a 2-depth nested list, each item is a list that contains boxes of the corresponding phrase
                            such as [[[295, 130, 366, 244], [209, 123, 300, 246], [347, 1, 439, 236]], [[0, 21, 377, 220]], [[0, 209, 214, 332]]]
                            there are three phrases, the first phrase contains three boxes while others contain only one box

Then, for one batch of data, the corresponding images_caption_phrase_bboxs is:
[ 
    [[[295, 130, 366, 244], [209, 123, 300, 246], [347, 1, 439, 236]], [[0, 21, 377, 220]], [[0, 209, 214, 332]]], - batch-1 
    [[[90, 68, 325, 374]], [[118, 64, 192, 128]]], 
    [[[1, 0, 148, 451]], [[153, 148, 400, 413]], [[374, 320, 450, 440]]]

]

'''


class F30KProvider(DatasetProviderROOT):
    ''' Dataset Provider for Flicker30Kentities
        
        Inputs:
            referitgame_base: an instance of the qmnist_base class

        Output:
            provide batch of images, labels = [digital, color_id, [ymin, xmin, ymax, xman]]
    '''

    def __init__(self, base_data, batch_size, epoch_size, num_data_loading_workers, 
                        transform_image_dec_func=None, transform_text_func=None, phase="train"):

        super().__init__(base_data, batch_size, epoch_size, num_data_loading_workers,
                        transform_image_dec_func,  transform_text_func, phase)      

        self.f30k_base = self.base_data

        # obtain the corresponding elements in this phase
        self.phase_data = self.f30k_base.get_phase_data(phase)
        self.phase_samples_name = list(self.phase_data.keys())
        
        self.f30k_base.print_phase_data_info(phase, self.phase_data)


    def get_one_sample(self, sample_idx):
        samle_retrieval_name = self.phase_samples_name[sample_idx]
        image_file_name = os.path.basename(samle_retrieval_name)
        image_id = os.path.splitext(image_file_name)[0]
        
        sample_name = image_id

        image_data = self.f30k_base.get_sample_image_data(self.phase, image_id)
        
        ori_image_data = image_data.copy()

        image_anno_sent = self.phase_data[samle_retrieval_name]

        sentence, sentence_phrases, sentence_phrases_type, \
            sentence_phrases_id, sentence_phrases_boxes = self.f30k_base.extract_sample_anno_data(image_anno_sent)

        # print("sentence: ", sentence)
        # print("sentence_phrases: ", sentence_phrases)
        # print("sentence_phrases_type: ", sentence_phrases_type)
        # print("sentence_phrases_id: ", sentence_phrases_id)
        # print("sentence_phrases_boxes: ", sentence_phrases_boxes)

        caption = caption if any(isinstance(iter_i, list) for iter_i in sentence) \
                                            else [[sentence]]
        flatten_caption_phrase_bboxs = [box for boxes in sentence_phrases_boxes for box in boxes]
        # ['The woman', 'mascara', 'the mirror']
        caption_phrases = [[phrase] for phrase in sentence_phrases]
        caption_phrases_cate = sentence_phrases_type
        caption_phrases_cate_id = sentence_phrases_id
        

 
        if self.transform_image_dec_func is not None:

            transformed = self.transform_image_dec_func(image=image_data, 
                                                        bboxes=flatten_caption_phrase_bboxs,
                                                        category_ids=range(len(flatten_caption_phrase_bboxs)))
            
            image_data = transformed["image"]
            image_data = torch.from_numpy(image_data)
            flatten_caption_phrase_bboxs = transformed["bboxes"]
            caption_phrase_bboxs = utils.phrase_boxes_alignment(flatten_caption_phrase_bboxs, sentence_phrases_boxes)

        if self.transform_text_func is not None:
            caption_phrase = self.transform_text_func(caption_phrase)


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
            elem_list=range(len(self.phase_samples_name)), load=load_function)
        
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_data_loading_workers),
            shuffle=(False if self.is_eval_mode else True),
            collate_fn=collate_fn)

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

        
    def __len__(self):
        return int(len(self.phase_samples_name) / self.batch_size)





