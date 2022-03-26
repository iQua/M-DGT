#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' python inherent libs '''
import os
import collections
import json
import collections
import sys
''' third parts libs '''
import numpy as np
''' Local libs '''
from .referMain import refer
'''
The referitgame dataset is different from other datasets because it does not have the 'actual' images or annotations files. 
We can get the idxs of imges for training, test, and val. Then, in each sample loading process, we get one idx and then extract the corresponding
'real' data.
'''

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
Splited_Datasets = collections.namedtuple('Splited_Datasets', [
    'train_ref_ids', 'val_ref_ids', 'test_ref_ids', 'testA_ref_ids',
    'testB_ref_ids', 'testC_ref_ids'
])


class REFERITGMBase(object):
    def __init__(self,
                 dataset_dir,
                 source_images_dir,
                 data_name="refcoco",
                 split_type="google"):
        self.dataset_dir = dataset_dir  # the source data directory of the original data
        self.source_images_dir = source_images_dir

        self.data_name = data_name

        self.split_type = split_type

        self._splited_referids_holder = dict()

        if data_name == "refcoco":
            if split_type not in ["google", "unc"]:
                print(
                    "************** The refcoco only supports 'google' or 'unc' as the split type *********"
                )
                sys.exit("A bad split type")

        if data_name == "refcoco+":
            if split_type != "unc":
                print(
                    "************** The refcoco+ only supports 'unc' as the split type *********"
                )
                sys.exit("A bad split type")

        if data_name == "refcocog":
            if split_type not in ["google", "umd"]:
                print(
                    "************** The refcocog only supports 'google' or 'umd' as the split type *********"
                )
                sys.exit("A bad split type")

        self._dataset_refer = refer.REFER(
            data_root=self.dataset_dir,
            image_dataroot=self.source_images_dir,
            dataset=self.data_name,
            splitBy=self.split_type)  # default is unc or google
        self._connect_to_splits()
        self._collect_data_info()

    def _connect_to_splits(self):
        split_types = Splited_Datasets._fields
        for split_type in split_types:
            formatted_split_type = split_type.strip().split("_")[0]
            self._splited_referids_holder[
                formatted_split_type] = self._dataset_refer.getRefIds(
                    split=formatted_split_type)

    def _collect_data_info(self):
        ''' Used to print the dataset_info '''
        print(
            "-" * 10 + "For current referitgame data (with " +
            self.split_type + "split" + ")", "we has created " +
            self.data_name + " provider" + " and the setting is:")
        for key in self._splited_referids_holder.keys():
            print(("-- {} with size: {}").format(
                key, len(self._splited_referids_holder[key])))

    def get_phase_data(self, phase):
        mode_refer_ids = self._splited_referids_holder[phase]

        mode_elements_holder = dict()
        mode_flatten_emelemts = list()

        for refer_id in mode_refer_ids:

            refer_elements = list()

            ref = self._dataset_refer.loadRefs(refer_id)[0]
            image_id = ref['image_id']
            image_file_path = self._dataset_refer.loadImgspath(image_id)
            caption_phrases_cate = self._dataset_refer.Cats[ref['category_id']]
            caption_phrases_cate_id = ref['category_id']

            mode_elements_holder[refer_id] = dict()
            mode_elements_holder[refer_id]["image_id"] = image_id
            mode_elements_holder[refer_id]["image_file_path"] = image_file_path

            mode_elements_holder[refer_id]["sentences"] = list()
            for send in ref["sentences"]:
                caption = send["tokens"]
                caption_phrase = send["tokens"]

                # images_data = dt_refer.loadImgData(image_id) # a list
                caption_phrase_bboxs = self._dataset_refer.getRefBox(
                    ref['ref_id'])  # [x, y, w, h]
                # convert to [xmin, ymin, xmax, ymax]
                caption_phrase_bboxs = [
                    caption_phrase_bboxs[0], caption_phrase_bboxs[1],
                    caption_phrase_bboxs[0] + caption_phrase_bboxs[2],
                    caption_phrase_bboxs[1] + caption_phrase_bboxs[3]
                ]

                sent_infos = {
                    "caption": caption,
                    "caption_phrase": caption_phrase,
                    "caption_phrase_bboxs": caption_phrase_bboxs,
                    "caption_phrases_cate": caption_phrases_cate,
                    "caption_phrases_cate_id": caption_phrases_cate_id
                }

                mode_elements_holder[refer_id]["sentences"].append(sent_infos)

                mode_flatten_emelemts.append([
                    image_id, image_file_path, caption, caption_phrase,
                    caption_phrase_bboxs, caption_phrases_cate,
                    caption_phrases_cate_id
                ])

        print("-" * 20 + "In phase: ", phase)
        print("-" * 10 + "phase holder has " + str(len(mode_elements_holder)) +
              " elemetents")
        print("-" * 10 + "phase holder has " +
              str(len(mode_flatten_emelemts)) + " flatten elemetents")
        return mode_elements_holder, mode_flatten_emelemts

    def get_image_data(self, image_id):
        image_data = self._dataset_refer.loadImgsData(image_id)[0]
        return image_data