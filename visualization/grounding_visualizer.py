#!/usr/bin/env python
# -*- coding: utf-8 -*-


''' Inherent Python '''
import os
import pickle
import re
import shutil
from collections import OrderedDict
import sys

''' Third Libs '''
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import scipy.misc
import pandas as pd
import tensorflow as tf
import six

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

''' custom libs '''
from .Configs import cv_colors_digial_map, cv_digital_colors_map, cv_colors_cmp_map_func
from .utils import numbering_directory, phrases_to_file, caption_to_file

''' Local Libs '''


class GroundingVisualizer(object):
    ''' Visualing the generated anchors and boxs '''

    def __init__(self, visualization_dir, is_unique_created=True):

        self._vis_log_dir = visualization_dir

        if is_unique_created:
            if not os.path.exists(self._vis_log_dir):
                os.makedirs(self._vis_log_dir, exist_ok=True)
            else:
                self._vis_log_dir = numbering_directory(self._vis_log_dir)
                os.makedirs(self._vis_log_dir)
        else:
            if not os.path.exists(self._vis_log_dir):
                raise FileNotFoundError("This directory is not existed, it should be created at first")


        self.reset_holder = self._vis_log_dir


    def set_save_dir(self, epoch_number, step_i):
        self._vis_log_dir = os.path.join(self._vis_log_dir, str(epoch_number) + "_" +str(step_i))
        os.makedirs(self._vis_log_dir, exist_ok=True)

    def reset_save_dir(self):
        self._vis_log_dir = self.reset_holder 

    
    def log_samples(self, images_name, original_images, processed_images, images_phrases, images_caption):
        """[save the original images and the processed images (inputs of the model)]

        Args:
            images_name ([list]): [each item is a string that shows the id of the image]
            original_images ([list]): [each item is a PIL object that can be saved diretly]
            processed_images ([list]): [each item is a torch tensor object]
            images_phrases ([list]): [each item is a list in which each item is the phrase of the image]
            images_caption ([list]): [each item is a list that contains the phrase of the corresponding image]
        """
        
        base_imgs_save_path = os.path.join(self._vis_log_dir, "samples")

        os.makedirs(base_imgs_save_path, exist_ok=True)


        def save_sample(img_idx):
            image_name = str(images_name[img_idx])
            ori_image_data = original_images[img_idx]
            prd_image_data = processed_images[img_idx] * 255.0
            image_phrases = images_phrases[img_idx]
            image_caption = images_caption[img_idx]
            cv2.imwrite(os.path.join(base_imgs_save_path, image_name + "_original.jpg"), ori_image_data) 

            image_data = prd_image_data if isinstance(prd_image_data, np.ndarray) else prd_image_data.numpy()
            result_image_data = image_data
            cv2.imwrite(os.path.join(base_imgs_save_path, image_name + "_input.jpg"), result_image_data) 

            phrases_to_file(image_phrases, 
                            to_save_file_path=os.path.join(base_imgs_save_path, image_name + "_phrases.txt"))
            caption_to_file(image_caption, 
                            to_save_file_path=os.path.join(base_imgs_save_path, image_name + "_caption.txt"))
        list(map(save_sample, range(len(images_name))))


    def log_images_boxes(self, images_name, processed_images, images_boxes, log_name="samples_boxes"):
        """[Log the images with boxes while making phrase as the text (replaced by using different colors for different phrases)]

        Args:
            images_name ([list]): [each item is a string that shows the id of the image,
                                    for example: images_name:  ['7196226216', '2824227447', '7800436386']]
            processed_images ([list]): [each item is a torch tensor object]
            images_boxes ([list]): [each item is the coordination of the boxes
                                    for example: images_boxes:
                                        [
                                            [[113, 71, 280, 329], [62, 58, 146, 245], [466, 65, 499, 240], [196, 63, 367, 331, [66, 289, 113, 332]], 
                                            [[229, 11, 401, 258], [297, 107, 393, 243]], 
                                            [[283, 13, 406, 371], [[289, 95, 400, 344], [35, 191, 120, 350], [244, 172, 299, 326], [152, 186, 247, 356], [100, 188, 161, 373], [0, 184, 98, 352], [420, 190, 470, 267], [303, 301, 499, 374]]
                                        ]. ]
        """
        base_imgs_boxes_save_path = os.path.join(self._vis_log_dir, log_name)

        os.makedirs(base_imgs_boxes_save_path, exist_ok=True)

   
        def save_sample(img_idx):
            image_name = str(images_name[img_idx])
            prd_image_data = processed_images[img_idx] * 255.0

            image_boxes = images_boxes[img_idx]

            self.visual_boxes(image_name=image_name, draw_board=prd_image_data, boxes=image_boxes, target_colors_mapper=cv_digital_colors_map, 
                            texts=None,
                            save_path=os.path.join(os.path.join(base_imgs_boxes_save_path, image_name + "_boxes.jpg")))
            
        list(map(save_sample, range(len(images_name))))
    

    def log_images_labled_itered_boxes(self, images_name, processed_images, iters_trans_results, log_name="iteration_boxes"):
        """[Log the images with boxes while making phrase as the text (replaced by using different colors for different phrases)]

        Args:
            images_name ([list]): [each item is a string that shows the id of the image,
                                    for example: images_name:  ['7196226216', '2824227447', '7800436386']]
            processed_images ([list]): [each item is a torch tensor object]
            iters_trans_results ([OrderDict]): [each item is the obtained results of i-th iteration
                                              iters_trans_results[iter_i] is an orderdict that contains:  
                                                generated_boxes: the array with shape <n_boxes, 4>
                                                boxes_appraoch_phrases_label: the array with shape <n_boxes, 4>
                                                boxes_appraoch_phrases_inner_gt_box: the array with shape <n_boxes, 4>
                                                boxes_appraoch_phrases_inner_gt_box_index: the array with shape <n_boxes>
                                                boxes_appraoch_phrases_iou: the array with shape <n_boxes>
                                        ]
        """
        # Note: this function default believes that there is only 1 image that is required to be processed
        base_imgs_boxes_save_path = os.path.join(self._vis_log_dir, log_name)

        os.makedirs(base_imgs_boxes_save_path, exist_ok=True)


        only_image_name = images_name[0]
        only_image_name = str(only_image_name) if isinstance(only_image_name, int) else only_image_name
        only_processed_image = processed_images[0] * 255.0
        
        total_removed_boxes = None
        for iter_i in list(iters_trans_results.keys()):
            if not isinstance(iter_i, int):
                continue
            iter_trans_results = iters_trans_results[iter_i]
            generated_boxes = iter_trans_results["generated_boxes"]
            boxes_appraoch_phrases_label = iter_trans_results["boxes_appraoch_phrases_label"]

            if iter_i > 0:
                remove_box_idex = iters_trans_results[iter_i-1]["remove_box_idex"]

                if total_removed_boxes is None:
                    total_removed_boxes = remove_box_idex
                else:
                    total_removed_boxes = np.concatenate([total_removed_boxes, remove_box_idex], axis=None)


                generated_boxes = np.delete(generated_boxes, total_removed_boxes, 0)
                boxes_appraoch_phrases_label = np.delete(boxes_appraoch_phrases_label, total_removed_boxes, 0)

            generated_boxes = generated_boxes.tolist()
            self.visual_labeled_boxes(image_name=only_image_name, draw_board=only_processed_image, boxes=generated_boxes, 
                                    boxes_label=boxes_appraoch_phrases_label[:, 0], 
                                    target_colors_mapper=cv_digital_colors_map,
                                    save_path=os.path.join(os.path.join(base_imgs_boxes_save_path, only_image_name + "_boxes_" + str(iter_i) + ".jpg")), 
                                    texts=None)
            # if iter_i > 1:
            #     break


    def log_images_boxes_with_label(self, images_name, processed_images, images_phrases, images_phrases_boxes,
                                    log_name="samples_boxes"):
        """[Log the images with boxes while making phrase as the text (replaced by using different colors for different phrases)]

        Args:
            images_name ([list]): [each item is a string that shows the id of the image,
                                    for example: images_name:  ['7196226216', '2824227447', '7800436386']]
            processed_images ([list]): [each item is a torch tensor object]
            images_phrases ([list]): [each item is a list that contains the phrase,
                                    for example: images_phrases:  
                                        [[['The white team'], ['the blue team'], ['the soccer ball']], 
                                        [['A boy'], ['a green shirt']], 
                                        [['A lady'], ['a short black dress'], ['a crowd of young people'], ['a stage']]]. ]
            images_phrases_boxes ([list]): [each item is the nested list that contains the phrase-wise boxes
                                    for example: images_phrases_boxes:
                                        [
                                            [[[113, 71, 280, 329], [62, 58, 146, 245], [466, 65, 499, 240]], [[196, 63, 367, 331]], [[66, 289, 113, 332]]], 
                                            [[[229, 11, 401, 258]], [[297, 107, 393, 243]]], 
                                            [[[283, 13, 406, 371]], [[289, 95, 400, 344]], [[35, 191, 120, 350], [244, 172, 299, 326], [152, 186, 247, 356], [100, 188, 161, 373], [0, 184, 98, 352], [420, 190, 470, 267]], [[303, 301, 499, 374]]]
                                        ]. ]
        """
        base_imgs_boxes_save_path = os.path.join(self._vis_log_dir, log_name)

        os.makedirs(base_imgs_boxes_save_path, exist_ok=True)
        def save_sample(img_idx):
            image_name = str(images_name[img_idx])
            prd_image_data = processed_images[img_idx] * 255.0
            image_phrases = images_phrases[img_idx]
            image_phrases_boxes = images_phrases_boxes[img_idx]

            self.visual_boxes_with_labels(image_name=image_name, draw_board=prd_image_data, boxes=image_phrases_boxes, 
                                 boxes_labels=list(range(len(image_phrases))), target_colors_mapper=cv_digital_colors_map, 
                                 texts=None,
                                 save_path=os.path.join(os.path.join(base_imgs_boxes_save_path, image_name + "_boxes.jpg")))
            
        list(map(save_sample, range(len(images_name))))
        

    def log_final_boxes(self, images_name, processed_images, images_phrases, 
                images_phrases_boxes, iters_trans_results,
                log_name="final_boxes"):

        base_imgs_boxes_save_path = os.path.join(self._vis_log_dir, log_name)

        os.makedirs(base_imgs_boxes_save_path, exist_ok=True)

        final_boxes_array = iters_trans_results["final_boxes"]
        final_boxes_score_array = iters_trans_results["final_boxes_score_array"]
        final_boxes_label = iters_trans_results["final_boxes_label"]

        def save_sample(img_idx):
            image_name = str(images_name[img_idx])
            prd_image_data = processed_images[img_idx] * 255.0
            image_phrases = images_phrases[img_idx]
            image_phrases_boxes = images_phrases_boxes[img_idx]


            new_board = self.visual_labeled_boxes(image_name=image_name, draw_board=prd_image_data, boxes=final_boxes_array.tolist(), 
                                        boxes_label=final_boxes_label, 
                                        target_colors_mapper=cv_digital_colors_map,
                                        save_path=os.path.join(os.path.join(base_imgs_boxes_save_path, "final_predicted.jpg")), 
                                        is_filled=True,
                                        texts=None, is_save=True)

            self.visual_boxes_with_labels(image_name=image_name, draw_board=new_board, boxes=image_phrases_boxes, 
                                 boxes_labels=list(range(len(image_phrases))), target_colors_mapper=cv_digital_colors_map, 
                                 texts=None,
                                 target_color="black",
                                 save_path=os.path.join(os.path.join(base_imgs_boxes_save_path, image_name + "_final_boxes.jpg")))
            
        list(map(save_sample, range(len(images_name))))


    def visual_boxes(self, image_name, draw_board, boxes, save_path, target_colors_mapper=cv_digital_colors_map, 
                        texts=None):
        """[Visualize the boxes of the image]

        Args:
            image_name ([string]): [the name of image]
            draw_board ([cv2 instance]): [the ]
            boxes ([list]): [a list, each item is a list that presents the boxes of the phrase,
                            such as [85.125, 0.0, 186.55312500000002, 153.8550724637681].]
            target_colors_mapper ([type], optional): [description]. Defaults to None.
            texts ([type], optional): [description]. Defaults to None.
            save_pre (str, optional): [description]. Defaults to 'VisualLabelTest'.
        """
        draw_board = draw_board if isinstance(draw_board, np.ndarray) else draw_board.numpy()
        draw_board_image = draw_board
        board_h, board_w = draw_board.shape[:2]
        alpha = 0.2  # Transparency factor.

        image_new = draw_board_image.copy()

        # label_idx = 11 # the label of all boxes are set to 0
        box_color_name = "blue"
        box_color_set = cv_colors_cmp_map_func(box_color_name)

        
        for box_coor in boxes:
            [x_min, y_min, x_max, y_max] = box_coor
            [x_min, y_min, x_max, y_max] = [int(x_min), int(y_min), int(x_max), int(y_max)]
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            if x_max > board_w:
                x_max = board_w
            if y_max > board_h:
                y_max = board_h  
            # Draw rectangles
            blk = image_new.copy()
            blk = cv2.rectangle(blk, (x_min, y_min), (x_max, y_max), box_color_set, cv2.FILLED)
            # Following line overlays transparent rectangle over the image
            image_new = cv2.addWeighted(blk, alpha, image_new, 1 - alpha, 0)

            thickness = 1 if x_max - x_min < 40 and y_max - y_min < 40 else 2
            image_new = cv2.rectangle(image_new, (x_min, y_min), (x_max, y_max), box_color_set, thickness=thickness)

        cv2.imwrite(save_path, image_new) 


    def visual_labeled_boxes(self, image_name, draw_board, boxes, boxes_label, save_path, target_colors_mapper=cv_digital_colors_map, 
                            texts=None, is_filled=False, is_save=True):
        """[Visualize the boxes of the image]

        Args:
            image_name ([string]): [the name of image]
            draw_board ([cv2 instance]): [the ]
            boxes ([list or array]): [a list, each item is a list that presents the boxes of the phrase,
                            such as [85.125, 0.0, 186.55312500000002, 153.8550724637681].]
            boxes_label (list or 1d array): [the corresponding label of the boxes].
            texts ([type], optional): [description]. Defaults to None.
        """
        draw_board = draw_board if isinstance(draw_board, np.ndarray) else draw_board.numpy()
        draw_board_image = draw_board
        board_h, board_w = draw_board.shape[:2]
        alpha = 0.2  # Transparency factor.

        image_new = draw_board_image.copy()

        for box_coor_i in range(len(boxes)):
            box_coor = boxes[box_coor_i]
            box_label = boxes_label[box_coor_i]
            box_color_name = target_colors_mapper[box_label]
            box_color_set = cv_colors_cmp_map_func(box_color_name)

            [x_min, y_min, x_max, y_max] = box_coor
            [x_min, y_min, x_max, y_max] = [int(x_min), int(y_min), int(x_max), int(y_max)]
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            if x_max > board_w:
                x_max = board_w
            if y_max > board_h:
                y_max = board_h   
            # Draw rectangles
            blk = image_new.copy()
            if is_filled:
                blk = cv2.rectangle(blk, (x_min, y_min), (x_max, y_max), box_color_set, cv2.FILLED)
            else:
                blk = cv2.rectangle(blk, (x_min, y_min), (x_max, y_max), box_color_set)
            # Following line overlays transparent rectangle over the image
            image_new = cv2.addWeighted(blk, alpha, image_new, 1 - alpha, 0)

            thickness = 1 if x_max - x_min < 40 and y_max - y_min < 40 else 2
            image_new = cv2.rectangle(image_new, (x_min, y_min), (x_max, y_max), box_color_set, thickness=thickness)

        if is_save:
            cv2.imwrite(save_path, image_new) 
        return image_new
        


    def visual_boxes_with_labels(self, image_name, draw_board, boxes, save_path,
                                 boxes_labels, target_colors_mapper=cv_digital_colors_map,
                                 target_color=None,
                                 texts=None):
        """[Visualize the boxes of the image]

        Args:
            image_name ([string]): [the name of image]
            draw_board ([cv2 instance]): [the ]
            boxes ([list]): [a 2-depth nested list, each item is a nested list that presents the boxes of the phrase,
                            such as [[(85.125, 0.0, 186.55312500000002, 153.8550724637681)]].]
            boxes_labels ([type], optional): [description]. Defaults to None.
            target_colors_mapper ([type], optional): [description]. Defaults to None.
            texts ([type], optional): [description]. Defaults to None.
            save_pre (str, optional): [description]. Defaults to 'VisualLabelTest'.
        """
        draw_board = draw_board if isinstance(draw_board, np.ndarray) else draw_board.numpy()
        draw_board_image = draw_board
        board_h, board_w = draw_board.shape[:2]

        alpha = 0.2  # Transparency factor.

        image_new = draw_board_image.copy()

        # print("boxes_labels: ", boxes_labels)
        # print("boxes: ", boxes)
        for label_idx in range(len(boxes_labels)):
            phrase_boxes = boxes[label_idx]

            if target_color is None:
                box_color_name = target_colors_mapper[label_idx]
                box_color_set = cv_colors_cmp_map_func(box_color_name)
            else:
                box_color_name = target_color
                box_color_set = cv_colors_cmp_map_func(box_color_name)

            for box_coor in phrase_boxes:
                [x_min, y_min, x_max, y_max] = box_coor
                [x_min, y_min, x_max, y_max] = [int(x_min), int(y_min), int(x_max), int(y_max)]

                if x_min < 0:
                    x_min = 0
                if y_min < 0:
                    y_min = 0
                if x_max > board_w:
                    x_max = board_w
                if y_max > board_h:
                    y_max = board_h   

                # Draw rectangles
                blk = image_new.copy()
                blk = cv2.rectangle(blk, (x_min, y_min), (x_max, y_max), box_color_set, cv2.FILLED)
                # Following line overlays transparent rectangle over the image
                image_new = cv2.addWeighted(blk, alpha, image_new, 1 - alpha, 0)

                thickness = 1 if x_max - x_min < 40 and y_max - y_min < 40 else 2
                image_new = cv2.rectangle(image_new, (x_min, y_min), (x_max, y_max), box_color_set, thickness=thickness)

        cv2.imwrite(save_path, image_new) 



    @property
    def save_visual_dir(self):
        return self._vis_log_dir
    

if __name__=="__main__":
    pass