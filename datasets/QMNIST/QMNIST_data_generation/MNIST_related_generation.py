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
from path_utils import SOURCES_p
from MNIST_base_generator import MNISTBaseGenerator


# the original MNIST dataset 
MNIST_data_path = os.path.join(SOURCES_p, 'datasets','CV_datasets', 'MNIST_Related', 'MNIST_data')

# the destination directory to saving the generated dataset
dest_path = os.path.join(SOURCES_p, 'datasets','CV_datasets', 'MNIST_Related', 'MNICSBT_data')



class MNISTDigitalQuesGenerator(MNISTBaseGenerator):
    ''' 

        Inputs:
            original_data_pt: the path of the original MNIST images
            generated_img_size: size of the genearted image (h, w)
            put_digital_scale: the zoom-scale of the digital we want to put in the image
            with_color: whether we use colored image (colored digital and query)
            color_noise_param: all noise to the color

    '''
    def __init__(self, original_data_pt, generated_img_size, put_digital_scale, with_color=True, color_noise_param=(0, 10)):

        super().__init__(original_data_pt, generated_img_size, put_digital_scale)
        self._digitals_pools = self._count_diginal_samples(self._itemset)

        if with_color:
            self._colors_map = self._get_color_maps()
            self._color_noise_param = color_noise_param
        else:
            self._colors_map = {'query': None}
            self._color_noise_param = None

        self._with_color = with_color

    def __extract_query_patch(self, digital_num, color_name=None):
        ''' extracting one patch image from all the images of digital_num '''
        cur_digital_imgs = self._digitals_pools[digital_num] #
        selected_digital_img_idx= np.random.choice(len(cur_digital_imgs), 1, replace=False)[0]

        sampled_digital_img = cur_digital_imgs[selected_digital_img_idx][1]

        if color_name is None:
            # the extracting image is the selectd image directly
            newImg = sampled_digital_img
            newImg = newImg.astype(np.uint8)
            extract_query_image = image.fromarray(newImg)

        else:
            # we extracting image with specific color based on the selected image
            (h, w) = sampled_digital_img.shape
            newImg = np.zeros((w, h, 4), dtype=np.uint8)
            color = self._colors_map[color_name]
            for i in range(w):
                for j in range(h):
                    if sampled_digital_img[i, j] != 0:
                        newImg[i, j, 0:3] = color *(sampled_digital_img[i, j]) / 255.
                        newImg[i, j, 3] = sampled_digital_img[i, j]

                newImg[newImg > 255] = 255
                newImg[newImg < 0] = 0
                newImg = newImg.astype(np.uint8)
                img = image.fromarray(newImg)
                # img = image.fromarray(background)
                # img.paste(img_, None, img_)
                extract_query_image = img.convert('RGB')

        return extract_query_image

    def _create_sampled_query_patch(self, opt_des_dir):
        """ For each number from 0-9, generate patch for valization """
        for i in range(self._num_digitals):
            query_img_dir = opt_des_dir + '/questions/%d' % i
            if not os.path.exists(query_img_dir):
                os.makedirs(query_img_dir)

            if self._with_color:
                for name in self._colors_map.keys():
                    img = self.__extract_query_patch(i, name)
                    img.save(os.path.join(query_img_dir, "%s.png" % name))
            else:
                name = 'query'
                img = self.__extract_query_patch(i, None)
                img.save(os.path.join(query_img_dir, "%s.png" % name))


    def _query_info_each_image(self, minn, maxn, digital_repeat=False):
        ''' construct query information for one image 
            
            Args:
                minn: the minimum number of digitals in each image
                maxn: the maximum number of digital in eahc image
                digital_repeat: whether one digital can appear in the image with different color for many times
            Output:
                query_infos: a list contains series of array [query_digital, query_digital_color_name]
        '''
        query_infos = list()

        candidate_numbers = list(range(minn, maxn))
        num_queried_digitals = np.random.choice(candidate_numbers, 1, replace=False)[0]

        color_names = list(self._colors_map.keys())
        num_colors = len(color_names)

        if digital_repeat:
            threshold = num_colors * self._num_digitals
        else:
            threshold = self._num_digitals

        if num_queried_digitals > threshold:
            raise('The required number of queries is exceed the minimum size')

        # indicator_mat is the indicator of all the combination of digital and color
        # each element equals to 1 means that the corresponding combination is nor used
        indicator_mat = np.ones((self._num_digitals, num_colors)) #

        for info_idx in range(num_queried_digitals):
            useful_combinations_idx = np.transpose(np.nonzero(indicator_mat))
            selected_combination_idx = np.random.choice(len(useful_combinations_idx), 1, replace=False)[0]
            
            # [digital, color_idx]
            selected_combination = useful_combinations_idx[selected_combination_idx] 
            selected_digital = selected_combination[0]
            selected_color_idx = selected_combination[1]
            selected_color_name = color_names[selected_color_idx]

            query_infos.append((selected_digital, selected_color_name))
            if digital_repeat:
                indicator_mat[selected_digital, selected_color_idx] = 0
            else:
                indicator_mat[selected_digital, :] = 0

        return query_infos

    def _construct_image(self, img_idx, queries_info, dis_size = [5, 5], num_distor=10):
        ''' constructing each images according the question i
            
            img_idx: the idx of current image
            ququeries_infoery_info: the information of queried digital in this image
            dis_size: the base size of the distractors in one image
            num_distor: the number of distractor added to the image
        '''
        [h, w] = self._generated_img_sz
        generated_bboxs = list()

        background = self._generate_distraction(self._digitals_pools, dis_size, num_distor, self._with_color,
                                                colors_map=self._colors_map, color_nosie_param=self._color_noise_param)

        newImg, generated_bboxs, scales = self._generate_img(queries_info, self._digitals_pools, self._with_color,
                                                    colors_map=self._colors_map, color_nosie_param=self._color_noise_param)

        img_ = image.fromarray(newImg)
        img = image.fromarray(background)
        img.paste(img_, None, img_)

        if self._with_color:
            img = img.convert('RGB')

        return img, generated_bboxs, scales

    def generate_images_labels(self, total_num_images, to_svae_des_path, repeat_query = False, range_queries_pre_img = [4, 6], dis_size = [5, 5], num_distor=15):

        dest_path = to_svae_des_path
        labels = []
        total_scales = []
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        if not os.path.exists(dest_path + '/images'):
            os.makedirs(dest_path + '/images')
        if not os.path.exists(dest_path + '/images_bbox'):
            os.makedirs(dest_path + '/images_bbox')

        for i in range(total_num_images):
            img_idx = i
            if i % 1000 == 0:
                print('\r', i, end=' ')
            sys.stdout.flush()

            q_infos = self._query_info_each_image(minn=range_queries_pre_img[0], maxn=range_queries_pre_img[1])

            img, generated_bboxs, used_scales = self._construct_image(img_idx, q_infos, dis_size, num_distor)

            ori_img_path = dest_path + '/images/%05d.png' % img_idx
            bboxs_img_path = dest_path + '/images_bbox/%05d.png' % img_idx

            img.save(ori_img_path)

            # draw generated bboxs in the image
            img = img.convert('RGB') # we should draw the bboxs by the red line
            for coor in generated_bboxs:
                # the coor of the generatd bbox is [ymin, xmin, ymax, xmax]
                # here we should convert it to [xmin, ymin, xmax, ymax] to draw
                img_obj = ImageDraw.Draw(img)
                img_obj.rectangle((coor[1], coor[0], coor[3], coor[2]), outline="red")

            img.save(bboxs_img_path)

            total_scales.append(used_scales)
            for idx, item in enumerate(q_infos):
                labels.append(','.join(['%05d' % img_idx]
                                       + [str(x) for x in item[:]]
                                       + [str(coors) for coors in generated_bboxs[idx]]))

        labelOut = open(dest_path + '/labels.txt', 'wt')
        labelOut.write('\n'.join(labels))
        labelOut.close()
        scaleOut = open(dest_path + '/total_scales.txt', 'wt')
        scaleOut.write('\n'.join([str(x) for use_scale in total_scales for x in use_scale]))
        scaleOut.close()
        print()


class MNISTRelatedDSGenerator(MNISTBaseGenerator):
    ''' Using to genearting the MNIST-Related dataset 
        manily the following datasets:
            1. extension of the original datasets
            - original dataset
            - arbitrary size digital image with and without noise
            - arbitrary size colored digital image with and without noise

            2. question-related datasets
            - mnist-colored-questions dataset
            - mnist-RE-colored-question dataset
    '''
    def __init__(self, original_data_pt):
        pass




if __name__ == "__main__":
    print('testing')
    # test_ds_generator = MNISTDigitalQuesGenerator(MNIST_data_path, generated_img_size=[100, 100], put_digital_scale = (0.7, 1.5), 
    #                                               with_color=True, color_noise_param=(0, 10))
    # # test_ds_generator._create_sampled_query_patch(dest_path)
    # # infos = test_ds_generator._query_info_each_image(4, 10)
    # # print(infos)
    # test_ds_generator.generate_images_labels(total_num_images=100, range_queries_pre_img = [1, 2], to_svae_des_path=dest_path)

    # test_ds_generator = MNISTDigitalQuesGenerator(MNIST_data_path, generated_img_size=[100, 100], put_digital_scale = (0.9, 2), 
    #                                               with_color=False)
    # # test_ds_generator._create_sampled_query_patch(dest_path)
    # # infos = test_ds_generator._query_info_each_image(4, 10)
    # # print(infos)
    # test_ds_generator.generate_images_labels(total_num_images=150000, range_queries_pre_img = [1, 2], to_svae_des_path=dest_path)



    test_ds_generator = MNISTDigitalQuesGenerator(MNIST_data_path, generated_img_size=[100, 100], put_digital_scale = (0.8, 1.2), 
                                                  with_color=True)
    # test_ds_generator._create_sampled_query_patch(dest_path)
    # infos = test_ds_generator._query_info_each_image(4, 10)
    # print(infos)
    test_ds_generator.generate_images_labels(total_num_images=20000, to_svae_des_path=dest_path, range_queries_pre_img = [4, 6])






