#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Inherent libs '''
import os
from collections import defaultdict
''' Third libs '''
import numpy as np
import cv2
''' Local libs '''


def decode_annotations_file(annos_file_path, num_channels, colors_digial_map):
    with open(annos_file_path, 'r') as f:
        read_annos_data = f.readlines()

    global_annos_data = defaultdict(dict)
    for line_anno_dt in read_annos_data:
        splitd_line_annos = line_anno_dt.rstrip('\n').strip().split(',')
        img_number = splitd_line_annos[0]
        digital_num = int(splitd_line_annos[1])

        if "phrases" not in global_annos_data[img_number].keys():
            global_annos_data[img_number]["phrases"] = list()

        if "digitalColors" not in global_annos_data[img_number].keys():
            global_annos_data[img_number]["digitalColors"] = list()

        if "digitalBoxes" not in global_annos_data[img_number].keys():
            global_annos_data[img_number]["digitalBoxes"] = list()

        if "digitalNumbers" not in global_annos_data[img_number].keys():
            global_annos_data[img_number]["digitalNumbers"] = list()

        if num_channels == 1:
            digital_bbox = list(map(lambda x: int(x), splitd_line_annos[2:]))
            global_annos_data[img_number]["phrases"].append(str(digital_num))
            global_annos_data[img_number]["digitalNumbers"].append(digital_num)
            global_annos_data[img_number]["digitalBoxes"].append(digital_bbox)
        else:
            color_str = splitd_line_annos[2]
            digital_color = colors_digial_map[color_str]

            global_annos_data[img_number]["phrases"].append(color_str + " " +
                                                            str(digital_num))

            digital_bbox = list(map(lambda x: int(x), splitd_line_annos[3:]))
            global_annos_data[img_number]["digitalNumbers"].append(digital_num)
            global_annos_data[img_number]["digitalColors"].append(
                digital_color)
            global_annos_data[img_number]["digitalBoxes"].append(digital_bbox)

    for img_number in list(global_annos_data.keys()):
        img_phrases = global_annos_data[img_number]["phrases"]
        caption = ' '.join(img_phrases)
        global_annos_data[img_number]["caption"] = [caption]

    return global_annos_data


def extract_image(img_f_path, num_channels):

    if num_channels == 1:
        im = cv2.imread(img_f_path,
                        cv2.IMREAD_GRAYSCALE)  # h x w with shape uint32
    else:
        im = cv2.imread(img_f_path,
                        cv2.IMREAD_COLOR)  # h x w x 3 with shape uint32

    if im.ndim == 2:  # we add 1 as channel in the third axis
        im = np.expand_dims(im, axis=2)
    return im


def align_img_label(imgs_f, global_labels_data):
    used_imgs = list()
    used_labels = list()
    for img_f_name in imgs_f:
        img_number = img_f_name.strip().split('.')[0]
        img_corres_labels = global_labels_data[img_number]
        for img_label in img_corres_labels:
            used_imgs.append(img_f_name)
            used_labels.append(img_label)

    return used_imgs, used_labels
