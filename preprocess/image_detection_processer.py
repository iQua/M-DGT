#!/usr/bin/env python
# -*- coding: utf-8 -*-




''' Inherent libs '''
import os


''' Third libs '''
import numpy as np
import torchvision.transforms as transforms
import albumentations as A
import cv2

''' Local libs '''



class ImageDetectionProcessor(object):
    def __init__(self, dataset_name, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        
        # here we just normalize the image to [0, 1] 
        #   the mean and std will be added in the GeneralizedRCNNTransform part
        self._normalize = A.Normalize(mean=mean, std=std)

    def create_image_dec_processor(self, resize_shape=[], target_shape=[], phase="train", randomization=True, normalization=True):
        process_functions = list()

        [resize_h, resize_w] = resize_shape
        [target_h, target_w] = target_shape

        if resize_h is not None:
            resize_func = A.Resize(height=resize_h, width=resize_w)
            process_functions += [resize_func]

        if phase == "train":
            if randomization is True:
                process_functions = process_functions + [A.RandomCrop(width=target_h, height=target_w), 
                                                        A.RandomBrightnessContrast(p=0.2),
                                                        A.HorizontalFlip()]
            else:
                if target_h is not None:
                    resize_func = A.Resize(height=target_h, width=target_w)
                    process_functions += [resize_func]
                    
        else:
            print("----Working phrase: ", phase)
            process_functions = process_functions + [A.Resize(height=resize_h, width=resize_w)]

        #process_functions.append(A.ToTensor())


        if normalization is True:
            process_functions.append(self._normalize)
        

        transform_funcs = A.Compose(process_functions, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        
        return transform_funcs
    
    



