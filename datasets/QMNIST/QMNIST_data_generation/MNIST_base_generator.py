#!/usr/bin/env python
# -*- coding: utf-8 -*-


''' python inherent libs '''
import sys
import os
import struct
import operator as op
import itertools

''' third parts libs '''
import numpy as np
from numpy.random import *
from scipy.misc import *
import operator as op
from dataset_utils import read
from PIL import Image as image
from PIL import ImageDraw
''' custom libs '''



class MNISTBaseGenerator(object):
    def __init__(self, original_data_pt, generated_img_size, put_digital_scale):
        self._origianl_mnist_ds = original_data_pt
        self._generated_img_sz = generated_img_size
        self._put_digital_scale = put_digital_scale # the range of scale used to resize the raw digitail image

        self._itemset = sorted([x for x in read(dataset="training", path=original_data_pt)] +
                                [x for x in read(dataset="testing", path=original_data_pt)], key=op.itemgetter(0))
        self._num_digitals = 10

        self._MNIST_sample_sz = [28, 28]

    def _count_diginal_samples(self, itemset):
        ''' getting the iamges for each digital 0~9 '''
        # there should have 10 elements in the digitals_pools
        # each element i in it contains all the iamges of the corresponding digital i 
        # type [...[(digitalnum, image_data), ...], ...]
        # digitals_pools[i] is a list with each element is a touple (digital_num, img_data)
        digitals_pools = [] 
        for i in range(9):
            count = 0
            for j in range(len(itemset)):
                if itemset[j][0] != i:
                    break
                count += 1
            digitals_pools.append(itemset[:count])
            itemset = itemset[count:]
        digitals_pools.append(itemset)

        return digitals_pools

    def _get_color_maps(self):

        colorNames = ['red', 'green', 'blue', 'yellow', 'white']
        colorMap = {'red': [0x92, 0x10, 0x10], 'green': [0x10, 0xA6, 0x51], 'blue': [0x10, 0x62, 0xC0], 
                    'yellow': [0xFF, 0xF1, 0x4E], 'white': [0xE0, 0xE0, 0xE0]} # for R G B channels 

        for k in list(colorMap.keys()):
            colorMap[k] = np.array(colorMap[k])

        return colorMap

    def _generate_distraction(self, digitals_pools, dis_size, num_dist, with_color=True, colors_map=None, color_nosie_param=None):
        ''' generate distractors for each image, the distractors are put in the background image'''
        dis_scale = self._put_digital_scale
        [w, h] = self._generated_img_sz

        MNIST_sample_sz = self._MNIST_sample_sz

        background = None
        if with_color:
            background = np.zeros((w, h, 4), dtype=np.uint8)
            background[..., 3] = 255
        else:
            background = np.zeros((w, h), dtype=np.uint8)

        for n in range(num_dist):
            # we crop distraction from each patch (which is 28 x 28 in MNIST)
            crop_distraction_pos = (int(uniform() * (MNIST_sample_sz[0] - dis_size[0])),
                                    int(uniform() * (MNIST_sample_sz[1] - dis_size[1])))


            used_digital = int(uniform() * 10) # use a random digital
            used_digital_imgs = digitals_pools[used_digital]
            used_digital_img_idx = int(uniform() * len(used_digital_imgs))

            # randomly select a image, then crop one piece area from it as the distractor
            patch_img_data = used_digital_imgs[used_digital_img_idx][1]
            croped_dis = patch_img_data[crop_distraction_pos[0]:crop_distraction_pos[0] + dis_size[0], 
                                        crop_distraction_pos[1]:crop_distraction_pos[1] + dis_size[1]]

            # resize the cropped piece according to the parammeter
            scale = float(uniform(dis_scale[0], dis_scale[1]))
            size = (int(dis_size[0] * scale), int(dis_size[1] * scale))
            croped_dis = imresize(croped_dis, size)

            # find a position to put the cropped distractor
            pos = np.round(np.array([uniform(0, w - size[0]), uniform(0, h - size[1])]))
            pos = pos.astype(np.uint8)  # [height, width]
            if with_color:
                colors_name = list(colors_map.keys())
                num_colors = len(colors_name)
                color = colors_map[colors_name[int(uniform() * num_colors)]]
                color_noise = normal(color_nosie_param[0], color_nosie_param[1])
                color = color + color_noise
                color[color < 0] = 0
                color[color > 255] = 255

                # put the distractor in the background 
                background[pos[0]:pos[0] + size[0], pos[1]:pos[1] + size[1],
                           0: 3] += (croped_dis[:, :, None] * color[None, None, :] / 255.).astype('uint8')
                

            else:
                background[pos[0]:pos[0] + size[0], pos[1]:pos[1] + size[1]] += croped_dis

        return background

    def _generate_img(self, queries_info, digitals_pools, with_color=True, colors_map=None, color_nosie_param=None):
        ''' generate image according to the query 

            return:
                newImg: the generated image with [h, w, 3] if colored  and [h, w] otherwise
                generated_bboxs: a list of bboxs for all digitals in this image, [ymin, xmin, ymax, xmax]
                scales: the used scale for zooming each digital

        '''
        resize_scale = self._put_digital_scale
        [w, h] = self._generated_img_sz
        MNIST_sample_sz = self._MNIST_sample_sz
        generated_bboxs = list()
        scales = list()

        if with_color:
            newImg = np.zeros((w, h, 4), dtype=np.uint8)
        else:
            newImg = np.zeros((w, h), dtype=np.uint8)


        for query_info in queries_info:
            (selected_digital, selected_color_name) = query_info

            selected_digital_imgs = digitals_pools[selected_digital]
            selected_img_idx = int(uniform() * len(selected_digital_imgs))

            # print mnistSample
            if with_color:
                color_noise = normal(color_nosie_param[0], color_nosie_param[1])
                color = colors_map[selected_color_name] + color_noise
                color[color < 0] = 0
                color[color > 255] = 255

            # used to overcome the overlap of the queried digital
            # so that we can have a reasonable positon to put our digital instead of make
            # several digital existed in same palce
            while True:
                scale = float(uniform(resize_scale[0], resize_scale[1]))
                size = (int(MNIST_sample_sz[0] * scale), int(MNIST_sample_sz[1] * scale))

                pos = np.round(np.array([uniform(0, w - size[0]), uniform(0, h - size[1])]))
                pos[pos < 0] = 0
                pos[pos > w - size[0]] = w - size[0]
                pos = pos.astype(np.uint8)

                found = True
                for i in range(size[0]):
                    for j in range(size[1]):
                        if with_color:
                            used_pos_cond = (newImg[pos[0] + i, pos[1] + j, :] != 0).any()
                        else:
                            used_pos_cond = (newImg[pos[0] + i, pos[1] + j] != 0).any()
                        if used_pos_cond:
                            found = False
                            break
                    if not found:
                        break
                if found:
                    break

            scales.append(scale)
            mnist_sample = imresize(selected_digital_imgs[selected_img_idx][1], scale)

            for i in range(size[0]):
                for j in range(size[1]):
                    if mnist_sample[i, j] != 0:
                        if with_color:
                            newImg[pos[0] + i, pos[1] + j, 0: 3] = color * (mnist_sample[i, j]) / 255.
                            newImg[pos[0] + i, pos[1] + j, 3] = mnist_sample[i, j]
                        else:
                            newImg[pos[0] + i, pos[1] + j] = mnist_sample[i, j]

            [ymin, xmin, ymax, xmax] = [pos[0], pos[1], pos[0] + size[0], pos[1] + size[1]]
            generated_bboxs.append([ymin + 2, xmin + 2, ymax - 2, xmax - 2])

        newImg[newImg > 255] = 255
        newImg[newImg < 0] = 0
        newImg = newImg.astype(np.uint8)



        return newImg, generated_bboxs, scales

