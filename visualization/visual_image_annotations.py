#!/usr/bin/env python
# -*- coding: utf-8 -*-



''' python inherent libs '''
import os
from collections import OrderedDict
''' third parts libs '''
import matplotlib as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import scipy.misc

''' local custom libs '''

from visualization.utils import convert_phrases_annotation, is_invaild_bboxs
from visualization.Configs import colors_digial_map, digital_colors_map, COLOR

def draw_image_boxs(image_board, box_coordinates, color, **kwargs):
    '''
    Draw the box  in the image
    Args:
        image_board: Image data, the image needed to be drawed. with shape [width, height]
        box_coordinates: list, each element is a list which containing the boudning
                            box coordinations([xmin, ymin, xmax, ymax]) of the input phrase
        color: string, the color of the bounding box
    '''
    def draw_single_box(image_b, coordinate, **sub_kwargs):
        drawObject = ImageDraw.Draw(image_b)

        if "outline" in sub_kwargs:
            drawObject.rectangle(coordinate, outline=sub_kwargs['outline'])
        elif "fill" in sub_kwargs:
            drawObject.rectangle(coordinate, fill=sub_kwargs['fill'])
        else:
            drawObject.rectangle(coordinate, fill='#FF0000')

        if "text" in sub_kwargs:
            text = sub_kwargs['text']
            try:
                text_color = sub_kwargs['text_color']
            except:
                text_color = '#FF0000'
            Font = ImageFont.truetype(os.path.join(os.getcwd(), '../visualization/font/timesbd.ttf'), 10, index=0)
            text_start_posx = coordinate[0]
            text_start_posy = coordinate[1] + 3
            text_w, text_h = Font.getsize(text)
            text_upper_left = (text_start_posx, text_start_posy)
            text_upper_right = (text_start_posx + text_w, text_start_posy + text_h)
            drawObject.rectangle(text_upper_left + text_upper_right, fill="#FFFFFF")
            drawObject.text(text_upper_left, text, fill=text_color, font=Font)
            # drawObject.rectangle(upper_left + bottom_right, outline=(255, 0, 0))
        return image_b

    color = COLOR[color]

    w, h = image_board.size

    img0 = image_board.copy()
    for box_coordinate in box_coordinates:
        img0 = draw_single_box(img0, box_coordinate, outline=color, text='')
        image_board = draw_single_box(image_board, box_coordinate, fill=color)

    img1 = img0.convert('RGBA')
    img2 = image_board.convert('RGBA')

    img_final = Image.blend(img1, img2, 0.4)
    img_final = img_final.convert('RGB')
    return img_final


def visual_image_annotations(images_id, images_data,\
                             captions_phrases=None, captions_phrases_bboxs=None,
                             captions_phrases_bboxs_label=None, captions_phrases_cate=None,
                             visual_save_dir=None):
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
        captions_phrases: Numpy with shape [batch_size, num_phrases, max_length_phrase]
        captions_phrases_bboxs: Numpy with shape [batch_size, num_total_bboxs, 4]
        captions_phrases_bboxs_label: Numpy with shape [batch_size, num_total_bboxs, 1]
                                        xmin, ymin, xmax, ymax
        captions_phrases_cate: Numpy with shape [batch_size, num_total_bboxs, 1]
        visual_save_dir: string, the path that used to save the visual result
    '''
    for idx in range(len(images_id)):
        try:
            image_id = images_id[idx].decode()
        except AttributeError:
            image_id = images_id[idx]
        #print("image ID: ", image_id)
        image_data = images_data[idx]
        # print(image_data.shape)


        # rows x cols x RGB
        img = scipy.misc.toimage(image_data)
        img_array = np.array(img)
        img = Image.fromarray(img_array)
        img.save(os.path.join(visual_save_dir, image_id + '.jpg'))

        if captions_phrases is not None:
            try:
                caption_phrases = [[word.decode() for word in ph if word.decode() != "NA"]
                                   for ph in captions_phrases[idx][:]]
            except AttributeError:
                caption_phrases = [[word for word in ph if word != "NA"]
                                   for ph in captions_phrases[idx][:]]     
            # print(caption_phrases)

            caption_phrases_bboxs = captions_phrases_bboxs[idx]
            caption_phrases_bboxs_label = captions_phrases_bboxs_label[idx].tolist()
            caption_phrases_cate = captions_phrases_cate[idx]

            phrases_bboxs, \
                phrases_bboxs_label,\
                phrases_cate = convert_phrases_annotation(caption_phrases_bboxs,
                                                          caption_phrases_bboxs_label,
                                                          caption_phrases_cate)
            # print(phrases_bboxs)
            # print(phrases_bboxs_label)
            # print(phrases_cate)

            img_board = img.copy()
            color_idx = 0
            for bbox_idx in range(len(phrases_bboxs_label)):
                bbox_label = phrases_bboxs_label[bbox_idx]
                bbox_coordinates = phrases_bboxs[bbox_idx]
                bbox_cate = phrases_cate[bbox_idx]
                if bbox_cate is "notvisual":
                    continue
                if is_invaild_bboxs(bbox_coordinates):
                    continue
                img_board = draw_image_boxs(img_board,
                                            bbox_coordinates,
                                            list(COLOR.keys())[color_idx])
                color_idx += 1
            img_board.save(os.path.join(visual_save_dir, image_id + '_bboxs.jpg'))
