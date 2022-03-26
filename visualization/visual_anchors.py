#!/usr/bin/env python
# -*- coding: utf-8 -*-



''' python inherent libs '''
import os
from collections import OrderedDict
import math

''' third parts libs '''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import scipy.misc

import skimage
from skimage import transform
''' local custom libs '''

from visualization.utils import convert_phrases_annotation, is_invaild_bboxs, overlap_fp_ROIS_stata

'''
Although we can have batch of images,\
Currently, we cannot handle the situation in which there are more that 1 batcdes

Therefore, the generated anchors are prepared for only one image (I.e. 1 batch), 

'''

def visual_image_anchors(images_id, images_data,
                         generated_anchors, visual_save_dir):
    '''
    Note: Since the data of each iamge in the batch are stored in array of numpy,
            they must share the same shape which means the sub-array in the array should
            be filled into the max size in the data.
    Visual the input and the corresponding annotations
    This function can be used to
    1. test the process of loading original data
    2. visual the model results

    Args:
        images_id: list, each element contained is the file number of the image
        images_data: Numpy with shape [batch_size, h, w, c]
        generated_anchors: Numpy with shape [fp_h*fp_w, 4] [x_min, y_min, x_max, y_max]
        visual_save_dir: string, the path that used to save the visual result
    '''
    # convert
    for idx in range(len(images_id)):
        try:
            image_id = images_id[idx].decode()
        except AttributeError:
            image_id = images_id[idx]

        #print("image ID: ", image_id)
        image_data = images_data[idx]

        # rows x cols x RGB
        img = scipy.misc.toimage(image_data)
        img_array = np.array(img)
        img = Image.fromarray(img_array)
        # img.save(os.path.join(visual_save_dir, image_id + '.jpg'))

        img_board = img.copy()

        drawObject = ImageDraw.Draw(img_board)

        for box_coordinate in generated_anchors:
            #print(box_coordinate)
            drawObject.rectangle(box_coordinate.tolist(), outline="red")

        img_board.save(os.path.join(visual_save_dir, image_id + '_anchors.jpg'))


def visual_image_anchors_attention(images_id, images_data, captions_phrases, iters_generated_anchors, iters_mapped_rois,
                         iters_iou_scores, embedded_images_fp, visual_save_dir):
    """ Visualize the generated anchors as attention hot map 
    
    Args:
        captions_phrases: Numpy with shape [batch_size, num_phrases, max_length_phrase], 
        iters_generated_anchors: Numpy with shape [num_iters, num_anchors, 4], coordinates for all anchors [x_min, y_min, x_max, y_max]
                                i-th item in this list contains the generated anchors for i-th iteration 
        iters_mapped_rois: Numpy with shape [num_iters, num_anchors, 4], The corresponding mapped ROI in required feature map
        iters_iou_scores: Numpy with shape [num_iters, num_phrases, num_anchors], each row (index by idx) contains the 
                            IOU score between the ground truth of idx phrase and all generated anchors
    """
    # display the visualization
    [_, embd_fp_h, embd_fp_w, _] = embedded_images_fp.shape



    def draw_image_attn_anchors(image_id, img, phrases, iters_generated_anchors, iters_mapped_rois, iters_iou_scores):

        [w, h] = img.size
        scales = [h/embd_fp_h, w/embd_fp_w]
        num_phrases = phrases.shape[0]
        num_iters = iters_generated_anchors.shape[0]
        up_scale = math.ceil(scales[0])

        smooth = True
        state_type = 'count'

        fig = plt.figure(1)

        if state_type is 'count':
            gs = gridspec.GridSpec(nrows=1, ncols=num_iters)

        elif state_type is 'value':
            gs = gridspec.GridSpec(nrows=num_iters, ncols=num_phrases)

        for ii_iter in range(num_iters):
            generated_anchors = iters_generated_anchors[ii_iter]
            mapped_ROIS = iters_mapped_rois[ii_iter]
            iou_score_ph = iters_iou_scores[ii_iter, :, :]

            pos_values = overlap_fp_ROIS_stata(mapped_ROIS, iou_score_ph, state_type=state_type, is_dis_weighted=True)
            
            if state_type is 'value':
                for i_ph in range(num_phrases):
                    ax_iter = fig.add_subplot(gs[ii_iter, i_ph])
                    ph = phrases[i_ph]

                    try:
                        ph_l = [p.decode() for p in ph.tolist()]
                    except AttributeError:
                        ph_l = [p for p in ph.tolist()]

                    ph = ' '.join([p for p in ph_l if p != 'NA'])
                    pos_value = pos_values[i_ph]
                    plt.imshow(img)

                    if ii_iter == 0:
                        ax_iter.set_title(ph, fontsize=10)
                    if smooth:
                        alpha_img = transform.pyramid_expand(pos_value, upscale=up_scale, sigma=2)
                    else:
                        alpha_img = transform.resize(ipos_value, [img.size[1], img.size[0]])
                    alpha_img = alpha_img[:h, :w]
                    plt.imshow(alpha_img, alpha=0.8)
                    plt.set_cmap(plt.cm.Greys_r)
                    plt.axis('off')

            else:
                # For all phrases, we only have one set of anchors, this is to say that the count density of anchors
                #   is the same for each phrase --> we only draw random one (set the first one) here

                general_count_state = pos_values[0]
                ax_iter = fig.add_subplot(gs[0, ii_iter])

                plt.imshow(img)
                if ii_iter == 0:
                    ax_iter.set_title("anchors density", fontsize=10)
                if smooth:
                    alpha_img = transform.pyramid_expand(general_count_state, upscale=up_scale, sigma=2)
                else:
                    alpha_img = transform.resize(general_count_state, [img.size[1], img.size[0]])             
                # alpha_img = alpha_img[:w, :h]
                alpha_img = alpha_img[:h, :w]
                plt.imshow(alpha_img, alpha=0.8)
                plt.set_cmap(plt.cm.Greys_r)
                plt.axis('off')

                
        plt.savefig(os.path.join(visual_save_dir, image_id + '_attn.jpg'))
        plt.close()

    for idx in range(len(images_id)):
        try:
            image_id = images_id[idx].decode()
        except AttributeError:
            image_id = images_id[idx]

        print("image ID: ", image_id)
        image_data = images_data[idx]
        phrases = captions_phrases[idx]
        # rows x cols x RGB
        img = scipy.misc.toimage(image_data)
        img_array = np.array(img)
        img = Image.fromarray(img_array)
        # img_array = np.array(img)
        # img = Image.fromarray(img_array)

        # img_board = img.copy()

        draw_image_attn_anchors(image_id, img, phrases, iters_generated_anchors, iters_mapped_rois, iters_iou_scores)








































