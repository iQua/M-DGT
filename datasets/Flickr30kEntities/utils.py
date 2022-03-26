#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Inherent libs '''
import os
import shutil
''' Third libs '''
import numpy as np
import xml.etree.ElementTree as ET
''' Local libs '''


def phrase_boxes_alignment(flatten_boxes, ori_phrases_boxes):
    phrases_boxes = list()

    ori_pb_boxes_count = list()
    for ph_boxes in ori_phrases_boxes:
        ori_pb_boxes_count.append(len(ph_boxes))

    strat_point = 0
    for pb_boxes_num in ori_pb_boxes_count:
        sub_boxes = list()
        for i in range(strat_point, strat_point + pb_boxes_num):
            sub_boxes.append(flatten_boxes[i])

        strat_point += pb_boxes_num
        phrases_boxes.append(sub_boxes)

    pb_boxes_count = list()
    for ph_boxes in phrases_boxes:
        pb_boxes_count.append(len(ph_boxes))

    assert pb_boxes_count == ori_pb_boxes_count

    return phrases_boxes


def filter_bad_boxes(boxes_coor):
    filted_boxes = list()
    for box_coor in boxes_coor:
        [xmin, ymin, xmax, ymax] = box_coor
        if xmin < xmax and ymin < ymax:
            filted_boxes.append(box_coor)

    return filted_boxes


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def list_inorder(listed_files, flag_str):
    filtered_listed_files = [fn for fn in listed_files if flag_str in fn]
    listed_files = sorted(filtered_listed_files,
                          key=lambda x: x.strip().split(".")[0])
    return listed_files


def copy_files(src_files, dst_dir):
    for file in src_files:
        shutil.copy(file, dst_dir)


def union_shuffled_lists(src_lists):
    for i in range(1, len(src_lists)):
        assert len(src_lists[i]) == len(src_lists[i - 1])
    p = np.random.permutation(len(src_lists[0]))

    return [np.array(ele)[p] for ele in src_lists]
