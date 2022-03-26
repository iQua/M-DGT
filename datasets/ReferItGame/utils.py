#!/usr/bin/env python
# -*- coding: utf-8 -*-


''' Inherent Python '''
import os
import re
import xml
import xml.dom.minidom as mindom
from collections import defaultdict


''' Third Libs '''
import numpy as np

''' Local Libs '''



def is_phrases_id_digit(phrases_id):
    for caption_phrases_id in phrases_id:
        for caption_phrase_id in caption_phrases_id:
            if not caption_phrase_id[0].isdigit():
                return False

    return True


def extract_info(contents):
    captions = list()
    phrases = list()
    phrases_id = list()
    phrases_cate = list()

    for content in contents:
        elem = re.findall(r"EN#(.+?)]", content)
        phrases.append([re.findall(r"\s(.+?)$", e) for e in elem])
        phrases_id.append([re.findall(r"^(.+?)/", e) for e in elem])
        phrases_cate.append([re.findall(r"/(.+?)\s", e) for e in elem])

        stad_caption = re.sub(pattern=r"\[(.+?)\s", repl="", string=content)
        stad_caption = re.sub(pattern=r"]", repl="", string=stad_caption)
        stad_caption = re.sub(pattern=r"\n", repl="", string=stad_caption)
        captions.append(stad_caption)

    assert len(phrases) == len(phrases_id) == len(phrases_cate)
    assert is_phrases_id_digit(phrases_id)

    return [captions, phrases, phrases_id, phrases_cate]


def sentence_info_extract(sentence_path):
    '''
    Args:
        sentence_path: string, path of the corredponding sentence path
    Outputs:
        captions: list, each element is the caption of the image
        phrases: list, each element is a list which contains all the phrases in the sentence
        phrases_id: list, each element is a list which contains the number of the
                            corresponding phrase
        phrases_cate: list, each element is a list which contains the category of the phrase
    '''
    with open(sentence_path, "r") as f:
        contents = f.readlines()
        [captions, phrases, phrases_id, phrases_cate] = extract_info(contents)

    return [captions, phrases, phrases_id, phrases_cate]


def annotations_map(annotation_path):
    '''
    Args:
        annotation_path: string, path of the corredponding annotation path
    '''
    DOMTree = mindom.parse(annotation_path)
    collection = DOMTree.documentElement
    # collecting all the movies in the xml
    objects = collection.getElementsByTagName("object")

    def default_novisual():
        return [[0, 0, 0, 0]]
    id_to_bbox = defaultdict(default_novisual)
    # get the detailed information for each object
    for obj in objects:
        try:
            xmin = int(obj.getElementsByTagName("xmin")[0].childNodes[0].data)
            ymin = int(obj.getElementsByTagName("ymin")[0].childNodes[0].data)
            xmax = int(obj.getElementsByTagName("xmax")[0].childNodes[0].data)
            ymax = int(obj.getElementsByTagName("ymax")[0].childNodes[0].data)
        except IndexError:
            [xmin, ymin, xmax, ymax] = [-1, -1, -1, -1]

        bbox_name = obj.getElementsByTagName('name')
        for i in range(len(bbox_name)):
            for child in bbox_name[i].childNodes:
                bbox_id = child.data
                id_to_bbox.setdefault(bbox_id, [])
                id_to_bbox[bbox_id].append([xmin, ymin, xmax, ymax])
    return id_to_bbox


def image_entities_extract(sentence_path, annotation_path):
    '''
    Args:
        sentence_path: string, path of the corredponding sentence path
        annotation_path: string, path of the corresponding annotation path
    '''
    [captions, phrases, phrases_id, phrases_cate] = sentence_info_extract(sentence_path)

    id_to_bbox = annotations_map(annotation_path)
    anno_bboxs = list()
    anno_bboxs_label = list()
    anno_phrases_cate = list()

    for phrase_id_idx in range(len(phrases_id)):
        phrase_id = phrases_id[phrase_id_idx]
        phrase_cate = phrases_cate[phrase_id_idx]

        anno_bboxs.append([id_to_bbox[sub_ph_id[0]] for sub_ph_id in phrase_id])

        anno_bboxs_label.append([[int(sub_ph_id[0])] * len(id_to_bbox[sub_ph_id[0]])
                                 for sub_ph_id in phrase_id])

        anno_phrases_cate.append([[phrase_cate[sub_ph_id_idx][0]]
                                  * len(id_to_bbox[phrase_id[sub_ph_id_idx][0]])
                                  for sub_ph_id_idx in range(len(phrase_id))])

    return captions, phrases, anno_bboxs, anno_bboxs_label, anno_phrases_cate
