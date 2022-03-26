#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Inherent libs '''
import os
import json
''' Third libs '''
import numpy as np
import skimage.io as io
import cv2
''' Local libs '''
from .utils import create_dir, list_inorder, copy_files, union_shuffled_lists, filter_bad_boxes
from .f30k_utils import get_sentence_data, get_annotations
'''
An example:
    sentence:  A boy catcher in a helmet , stretching out his gloved hand for a catch .
    sentence_phrases:  ['A boy catcher', 'a helmet', 'his gloved hand', 'a catch']
    sentence_phrases_type:  [['people'], ['clothing'], ['bodyparts'], ['other']]
    sentence_phrases_id:  ['62671', '62676', '62673', '62673']
    sentence_phrases_boxes:  [[[8, 2, 493, 449]], [[122, 3, 260, 143]], [[360, 59, 492, 140], [360, 59, 492, 140]], [[360, 59, 492, 140]]]
'''


class F30KEBase(object):
    def __init__(self, dataset_dir, split_type="cvpr"):
        self.dataset_dir = dataset_dir  # the source data directory of the original data
        self.split_type = split_type

        self.images_path = os.path.join(self.dataset_dir, "flickr30k_images")
        self.annos_path = os.path.join(self.dataset_dir, "Flickr30kEntities",
                                       "Annotations")  # .xml
        self.sents_path = os.path.join(self.dataset_dir, "Flickr30kEntities",
                                       "Sentences")  # .txt

        self.base_split_path = os.path.join(self.dataset_dir, "data_splited")
        self.train_split_path = os.path.join(self.base_split_path, "train")
        self.test_split_path = os.path.join(self.base_split_path, "test")
        self.val_split_path = os.path.join(self.base_split_path, "val")

        self.splits_path = {
            "train": self.train_split_path,
            "test": self.test_split_path,
            "val": self.val_split_path
        }

    def split_F30KE_dataset(self, train_rate=0.6, test_rate=0.2):

        if os.path.exists(self.base_split_path):
            print(
                "-" * 20 +
                "No need to genearte the split again, it already exists in " +
                self.base_split_path)
            return

        print("-" * 20 + "Genearting splited data (train/test/val) in " +
              self.base_split_path)

        total_images = list_inorder(os.listdir(self.images_path),
                                    flag_str=".jpg")
        total_sentences = list_inorder(os.listdir(self.sents_path),
                                       flag_str=".txt")
        total_annotations = list_inorder(os.listdir(self.annos_path),
                                         flag_str=".xml")

        total_images_count = len(total_images)

        train_images_number = int(total_images_count * train_rate)
        test_images_number = int(total_images_count * test_rate)
        val_images_number = total_images_count - train_images_number - test_images_number

        split_config = {
            "train": {
                "num": train_images_number
            },
            "test": {
                "num": test_images_number
            },
            "val": {
                "num": val_images_number
            }
        }

        # shuffle all data in order to generate splits
        [total_images, total_sentences,
         total_annotations] = union_shuffled_lists(
             [total_images, total_sentences, total_annotations])
        # saveing the images and entities to the corresponding directory
        for split in ["train", "test", "val"]:
            # 0. create directory for the splited data
            path = self.splits_path[split]
            splited_images_path = create_dir(
                os.path.join(path, ("{}_images").format(split)))
            splited_sentences_path = create_dir(
                os.path.join(path, ("{}_sentences").format(split)))
            splited_annotations_path = create_dir(
                os.path.join(path, ("{}_annotations").format(split)))

            # 1. gettting the data
            split_data_numbers = split_config[split]["num"]

            images = total_images[:split_data_numbers]
            split_images_path = [
                os.path.join(self.images_path, img_f) for img_f in images
            ]
            sentences = total_sentences[:split_data_numbers]
            split_sentences_path = [
                os.path.join(self.sents_path, sent_f) for sent_f in sentences
            ]
            annotations = total_annotations[:split_data_numbers]
            split_annotations_path = [
                os.path.join(self.annos_path, anno_f) for anno_f in annotations
            ]

            # 2. saving the splited data into the target file
            copy_files(split_images_path, splited_images_path)
            copy_files(split_sentences_path, splited_sentences_path)
            copy_files(split_annotations_path, splited_annotations_path)
            total_images = total_images[split_data_numbers:]
            total_sentences = total_sentences[split_data_numbers:]
            total_annotations = total_annotations[split_data_numbers:]

        print("-" * 20 + "Done!")

    def align_anno_sent(self, image_sents, image_annos):
        """[align the items in annotations and sentences]

        Args:
            image_sents ([list]): [each itme is a dict that contains 'sentence', 'phrases']
            image_annos ([dict]): [contain 'boxes' - a dict presents the phrase_id: box]

        Return:
             aligned_items ([list]): [each itme is a dict that contains the sentence with the corresponding phrases information,
                                        there should have several items because for one image, there are 5 sentences.
                                        Sometimes, some sentences are useless, making the number of items less than 5]
        """
        aligned_items = list()  # each item is a dict
        for sent_info in image_sents:

            img_sent = sent_info["sentence"]
            img_sent_phrases = list()
            img_sent_phrases_type = list()
            img_sent_phrases_id = list()
            img_sent_phrases_boxes = list()
            for phrase_info_idx in range(len(sent_info["phrases"])):
                phrase_info = sent_info["phrases"][phrase_info_idx]

                phrase = phrase_info["phrase"]
                phrase_type = phrase_info["phrase_type"]
                phrase_id = phrase_info["phrase_id"]
                if phrase_id not in image_annos["boxes"].keys():
                    continue

                phrase_boxes = image_annos["boxes"][phrase_id]  # a nested list
                filted_boxes = filter_bad_boxes(phrase_boxes)
                if not filted_boxes:
                    continue

                img_sent_phrases.append(phrase)
                img_sent_phrases_type.append(phrase_type)
                img_sent_phrases_id.append(phrase_id)
                img_sent_phrases_boxes.append(filted_boxes)

            if not img_sent_phrases:
                continue

            items = dict()
            items["sentence"] = img_sent  # a string shows the sentence
            items[
                "sentence_phrases"] = img_sent_phrases  # a list that contains the phrases
            items[
                "sentence_phrases_type"] = img_sent_phrases_type  # a nested list that contains phrases type
            items[
                "sentence_phrases_id"] = img_sent_phrases_id  # a list that contains the phrases  id
            items[
                "sentence_phrases_boxes"] = img_sent_phrases_boxes  # a nested list that contains boxes for each phrase

            aligned_items.append(items)

        return aligned_items

    def integrate_data(self, split_wise=True, globally=True):
        """[Integrate the data into one json file that contains aligned annotation-sentence for each image]"""
        def operate_integration(images_name, images_sentences_path,
                                images_annotations_path):
            integrated_data = dict()
            for image_name_idx in range(len(images_name)):
                image_name = images_name[image_name_idx]
                image_sent_path = images_sentences_path[image_name_idx]
                image_anno_path = images_annotations_path[image_name_idx]

                image_sents = get_sentence_data(image_sent_path)
                image_annos = get_annotations(image_anno_path)

                aligned_items = self.align_anno_sent(image_sents, image_annos)
                if not aligned_items:
                    continue
                for item_idx in range(len(aligned_items)):
                    integrated_data[image_name +
                                    str(item_idx)] = aligned_items[item_idx]

            return integrated_data

        if split_wise:
            for split in ["train", "test", "val"]:
                path = self.splits_path[split]
                save_path = os.path.join(path, split + "_integrated_data.json")
                if os.path.exists(save_path):
                    print("-" * 10 + "Integrated " + split + " file, Existed!")
                    continue
                splited_images_path = os.path.join(path,
                                                   ("{}_images").format(split))
                splited_sentences_path = os.path.join(
                    path, ("{}_sentences").format(split))
                splited_annotations_path = os.path.join(
                    path, ("{}_annotations").format(split))
                split_images = list_inorder(os.listdir(splited_images_path),
                                            flag_str=".jpg")
                split_sentences = list_inorder(
                    os.listdir(splited_sentences_path), flag_str=".txt")
                split_annotations = list_inorder(
                    os.listdir(splited_annotations_path), flag_str=".xml")

                split_sentences_path = [
                    os.path.join(splited_sentences_path, sent)
                    for sent in split_sentences
                ]
                split_annotations_path = [
                    os.path.join(splited_annotations_path, anno)
                    for anno in split_annotations
                ]

                split_integrated_data = operate_integration(
                    split_images, split_sentences_path, split_annotations_path)
                with open(save_path, 'w') as outfile:
                    json.dump(split_integrated_data, outfile)

                print("-" * 20 + "Integration for " + split + "Done!")

        if globally:
            save_path = os.path.join(self.dataset_dir,
                                     "total_integrated_data.json")
            if os.path.exists(save_path):
                print("-" * 10 + "Integrated whole file, Existed!")
                return
            total_images = list_inorder(os.listdir(self.images_path),
                                        flag_str=".jpg")
            total_sentences = list_inorder(os.listdir(self.sents_path),
                                           flag_str=".txt")
            total_annotations = list_inorder(os.listdir(self.annos_path),
                                             flag_str=".xml")

            total_sentences_path = [
                os.path.join(self.sents_path, sent) for sent in total_sentences
            ]
            total_annotations_path = [
                os.path.join(self.annos_path, anno)
                for anno in total_annotations
            ]

            split_integrated_data = operate_integration(
                total_images, total_sentences_path, total_annotations_path)
            with open(save_path, 'w') as outfile:
                json.dump(split_integrated_data, outfile)
            print("-" * 20 + "Integration for the whole dataset, Done!")

    def get_phase_data(self, phase):
        path = self.splits_path[phase]
        save_path = os.path.join(path, phase + "_integrated_data.json")
        with open(save_path, 'r') as outfile:
            phase_data = json.load(outfile)
        return phase_data

    def print_phase_data_info(self, phase, phase_data):
        path = self.splits_path[phase]
        phase_images_path = os.path.join(path, ("{}_images").format(phase))
        phase_images = os.listdir(phase_images_path)
        print("-" * 20 + phase + " data information")
        print("-" * 10 + " total images: " + str(len(phase_images)))
        print("-" * 10 + " total samples: " + str(len(phase_data)))

    def get_sample_image_data(self, phase, image_id):
        # get the image data
        image_phase_path = os.path.join(self.base_split_path, phase,
                                        ("{}_images").format(phase))
        image_data = io.imread(
            os.path.join(image_phase_path,
                         str(image_id) + ".jpg"))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        return image_data

    def extract_sample_anno_data(self, image_anno_sent):
        sentence = image_anno_sent["sentence"]  # a string
        sentence_phrases = image_anno_sent["sentence_phrases"]  # a list
        sentence_phrases_type = image_anno_sent[
            "sentence_phrases_type"]  # a nested list
        sentence_phrases_id = image_anno_sent["sentence_phrases_id"]  # a list
        sentence_phrases_boxes = image_anno_sent[
            "sentence_phrases_boxes"]  # a nested list

        return sentence, sentence_phrases, sentence_phrases_type, sentence_phrases_id, sentence_phrases_boxes