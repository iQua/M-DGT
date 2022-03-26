#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' python inherent libs '''
import os
import collections
import json
''' third parts libs '''
import numpy as np
''' custom libs '''
from .utils import decode_annotations_file, extract_image
from .config import colors_digial_map, digital_colors_map

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class QMNISTBase(object):
    def __init__(self, dataset_dir, split_type="cvpr"):
        self.dataset_dir = dataset_dir  # the source data directory of the original data

        if split_type == "cvpr":
            [self.train_size_rate, self.test_size_rate,
             self.val_size_rate] = [0.6, 0.2, 0.2]

        self.images_dir = os.path.join(self.dataset_dir, "images")
        self.annotations_file_path = os.path.join(self.dataset_dir,
                                                  "annotations.txt")

        # splited_file_names
        self.qminst_train_info_file_path = os.path.join(
            self.dataset_dir, "QMNIST_train.json")
        self.qminst_test_info_file_path = os.path.join(self.dataset_dir,
                                                       "QMNIST_test.json")
        self.qminst_val_info_file_path = os.path.join(self.dataset_dir,
                                                      "QMNIST_val.json")

    def collect_data_info(self):
        images_files_name = [
            img_f for img_f in os.listdir(self.images_dir) if ".png" in img_f
        ]
        with open(self.annotations_file_path, "r") as anno_f:
            imges_annotations = [img_anno for img_anno in anno_f.readlines()]

        offset = 20
        print("-" * offset + "QMINST" + "-" * offset)
        print("-" * int(offset / 2) + "images count: " +
              str(len(images_files_name)))
        print("-" * int(offset / 2) + "boxes count: " +
              str(len(imges_annotations)))
        print("-" * int(offset / 2) + "boxes type: y_min, x_min, y_max, x_max")
        print("\n")

    def color_id_mapper(self):
        return colors_digial_map

    def get_phase_data(self, phase):
        if phase == "train":
            phase_data_info_path = self.qminst_train_info_file_path
        elif phase == "test":
            phase_data_info_path = self.qminst_test_info_file_path
        else:
            phase_data_info_path = self.qminst_val_info_file_path

        with open(phase_data_info_path, 'r') as fp:
            phase_data_info = dict(json.load(fp))

        return phase_data_info

    def decode_annos(self, phase_sample_annos):
        # decode the annotations into detailed information

        # phase_data_info should be an array with size digital_num, digital_color, *digital_bbox

        digital_nums = phase_sample_annos["digitalNumbers"]
        digital_color = phase_sample_annos["digitalColors"]
        digital_bboxs = phase_sample_annos["digitalBoxes"]
        pharses = phase_sample_annos["phrases"]
        caption = phase_sample_annos["caption"]
        return digital_nums, digital_color, digital_bboxs, pharses, caption

    def convert_boxes(self, sample_bboxes):
        for box_coor_idx in range(len(sample_bboxes)):
            box_coor = sample_bboxes[box_coor_idx]
            [y_min, x_min, y_max, x_max] = box_coor

            sample_bboxes[box_coor_idx] = [x_min, y_min, x_max, y_max]

        return sample_bboxes

    def set_boxes(self, phase_sample_annos, sample_bboxes):
        phase_sample_annos["digitalBoxes"] = sample_bboxes

        return phase_sample_annos

    def set_numbers(self, phase_sample_annos, sample_numbers):
        phase_sample_annos["digitalNumbers"] = sample_numbers

        return phase_sample_annos

    def set_colors(self, phase_sample_annos, sample_colors):
        phase_sample_annos["digitalColors"] = sample_colors

        return phase_sample_annos

    def set_phrases_caption(self, phase_sample_annos, phrases, caption):
        phase_sample_annos["phrases"] = phrases
        phase_sample_annos["caption"] = caption

        return phase_sample_annos

    def split_qmnist_datasets(self, num_channels=3, is_save=True):
        ''' read the data and divide all the data into three categories 

            Args:
                train_size_rate: size of the training data over the whole dataset

        '''
        if os.path.exists(self.qminst_train_info_file_path) and os.path.exists(
                self.qminst_test_info_file_path):
            print("-" * 10 + "splited already existed in: " + self.dataset_dir)
            with open(self.qminst_train_info_file_path, 'r') as q_tr:
                train_data_info = json.load(q_tr)

            with open(self.qminst_test_info_file_path, 'r') as q_te:
                test_data_info = json.load(q_te)

            with open(self.qminst_val_info_file_path, 'r') as q_val:
                val_data_info = json.load(q_val)
        else:
            raw_labels_dt_pt = self.annotations_file_path

            global_annos_data = decode_annotations_file(
                raw_labels_dt_pt, num_channels, colors_digial_map)

            images_file = list(global_annos_data.keys())
            images_count = len(images_file)

            tr_length = int(images_count * self.train_size_rate)
            train_imgs_f_name = images_file[0:tr_length]

            te_length = int(images_count * self.test_size_rate)
            test_imgs_f_name = images_file[tr_length:tr_length + te_length]

            eval_imgs_f_name = images_file[tr_length + te_length:]

            train_data_info = dict((img_num, global_annos_data[img_num])
                                   for img_num in train_imgs_f_name)
            test_data_info = dict((img_num, global_annos_data[img_num])
                                  for img_num in test_imgs_f_name)
            val_data_info = dict((img_num, global_annos_data[img_num])
                                 for img_num in eval_imgs_f_name)

            if is_save:
                with open(self.qminst_train_info_file_path, 'w') as q_tr:
                    json.dump(train_data_info, q_tr)

                with open(self.qminst_test_info_file_path, 'w') as q_te:
                    json.dump(test_data_info, q_te)

                with open(self.qminst_val_info_file_path, 'w') as q_val:
                    json.dump(val_data_info, q_val)

        print("-" * 10 + "Training samples: " + str(len(train_data_info)))
        print("-" * 10 + "Testing samples: " + str(len(test_data_info)))
        print("-" * 10 + "Eval samples: " + str(len(val_data_info)))

        return train_data_info, test_data_info, val_data_info
