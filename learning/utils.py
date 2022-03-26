#!/usr/bin/env python
# -*- coding: utf-8 -*-



''' Inherent libs '''
import os
import time
import re


''' Third libs '''
import numpy as np
import torch

''' Local libs '''



def integrate_phrase_boxes(images_phrases_boxes):
    """[Integrate the boxes of phrases to a tensor corresponding one image]

    Args:
        images_phrases_boxes (list(list(list))): [ one example:
            [
                [[[156, 15, 459, 260]], [[0, 0, 499, 384]], [[34, 134, 233, 238], [0, 196, 499, 384]], [[0, 113, 499, 384]]], -> batch-1
                [[[237, 96, 452, 374], [36, 69, 216, 374]], [[255, 8, 499, 285]]],  -> batch-2
                [[[122, 108, 222, 447]], [[114, 102, 174, 155]], [[134, 155, 228, 292]], [[164, 282, 225, 367]], [[139, 411, 203, 433], [159, 392, 217, 414]]]
            ]
            
    """
    integrated_images_boxes = [torch.stack([torch.Tensor(box) for image_phrase_boxes in image_phrases_boxes for box in image_phrase_boxes]) 
                                                for image_phrases_boxes in images_phrases_boxes]
    return integrated_images_boxes

def get_label_integerID_mapper(batch_label_texts, batch_labels_id_tensor):
    """[Obtain the label and integer mapper]

    Args:
        batch_label_texts ([list]): [a list with length batch_size * way * shot, each item is a list that contains 
                                        the text label (string) for the corresponding image]
        batch_labels_id_tensor ([torch.tensor]): [a tensor with shape - [1, tasks_per_batch * way * shot]]
    """
    batch_label_texts = [lb_txt[0] for lb_txt in batch_label_texts]
    batch_labels_ids = batch_labels_id_tensor.reshape(-1).tolist()
    id2label_mapper = dict(zip(batch_labels_ids, batch_label_texts))

    mapper = id2label_mapper.copy()
    label2id_mapper = dict(zip(batch_label_texts, batch_labels_ids))
    mapper.update(label2id_mapper)
    
    ordered_labels_text = list() # each index position corresponds the label id
    for idx in range(len(id2label_mapper.keys())):
        ordered_labels_text.append(id2label_mapper[idx])
    
    return mapper, ordered_labels_text

def numbering_directory(target_directory_path):
    target_directory_name = target_directory_path.split("/")[-1]
    parent_dir = os.path.abspath(os.path.join(target_directory_path, '..'))

    existed_dirs = [dir_name for dir_name in os.listdir(parent_dir) 
                        if os.path.isdir(os.path.join(parent_dir, dir_name)) and target_directory_name in dir_name]

    existed_dirs_number = [int(re.search(r'\d+', dir_name).group()) for dir_name in existed_dirs
                            if re.search(r'\d+', dir_name)]
    sorted_number = sorted(existed_dirs_number)
    if sorted_number:
        target_directory_number = sorted_number[-1] + 1
    else:
        target_directory_number = "0"

    tg_dir = os.path.join(parent_dir, target_directory_name+str(target_directory_number))
    return tg_dir


# images_caption_phrase_bboxs:  [[[[260.55, 130.07, 362.19, 466.39]]]]
def noise_add(ori_images_caption_phrase_bboxs):
    # print("images_caption_phrase_bboxs: ", images_caption_phrase_bboxs)
    # conver the x and y

    new_images_caption_phrase_bboxs = [[[[0, 0, 0, 0]]]]

    tg_box = ori_images_caption_phrase_bboxs[0][0][0].copy()
    new_images_caption_phrase_bboxs[0][0][0][0] = tg_box[1]
    new_images_caption_phrase_bboxs[0][0][0][1] = tg_box[0]
    new_images_caption_phrase_bboxs[0][0][0][2] = tg_box[3]
    new_images_caption_phrase_bboxs[0][0][0][3] = tg_box[2]

    return new_images_caption_phrase_bboxs


def noise_add_shift(ori_images_caption_phrase_bboxs):
    # print("images_caption_phrase_bboxs: ", images_caption_phrase_bboxs)
    # conver the x and y

    new_images_caption_phrase_bboxs = [[[[0, 0, 0, 0]]]]

    tg_box = ori_images_caption_phrase_bboxs[0][0][0].copy()
    w_ = tg_box[2] - tg_box[0]
    h_ = tg_box[3] - tg_box[1]
    new_images_caption_phrase_bboxs[0][0][0][0] = tg_box[0] + np.random.randint(-w_, w_)
    new_images_caption_phrase_bboxs[0][0][0][1] = tg_box[1] + np.random.randint(-w_, w_)
    new_images_caption_phrase_bboxs[0][0][0][2] = tg_box[2] + np.random.randint(-h_, h_)
    new_images_caption_phrase_bboxs[0][0][0][3] = tg_box[3] + np.random.randint(-h_, h_)

    return new_images_caption_phrase_bboxs
