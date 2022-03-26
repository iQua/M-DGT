#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy as np


import os
import re


import numpy as np




def phrases_to_file(image_phrases, to_save_file_path):
    """ save a list phrases of the image to the file """
    image_phrases = image_phrases if any(isinstance(phrases_i, list) for phrases_i in image_phrases) else [image_phrases]
    with open(to_save_file_path, 'w') as file:
        for phrase in image_phrases:
            if isinstance(phrase, int):
                phrase = [str(phrase)]
            if isinstance(phrase, str):
                phrase = [phrase]
            s = " ".join(map(str, phrase))
            file.write(s+'\n')


def caption_to_file(image_caption, to_save_file_path):
    """ save a list phrases of the image to the file """
    image_caption_s = image_caption[0][0]

    with open(to_save_file_path, 'w') as file:
        file.write(image_caption_s)


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




# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")



def convert_phrases_annotation(caption_phrases_bboxs,
                               caption_phrases_bboxs_label,
                               caption_phrases_cate):
    '''
    Converting the annotations of the phrase into the standard type
    That is, each unique phrase corresponds to several bboxs
    Args:
        caption_phrases_bboxs: array [num_total_bboxs, 4],
                        each row contains the coordinate of the corresponding bboxs
                        organized as xmin, ymin, xmax, ymax
        caption_phrases_bboxs_label: list, length of num_total_bboxs,
                        each element is the label of the corresponding bbox
        caption_phrases_cate: array, [num_total_bboxs, 1],
                        each row is the type of the correspondign bbox
    '''
    phrases_bboxs = list()
    phrases_bboxs_label = list()
    phrases_cate = list()

    unique_phrases_bboxs_label = list()
    for label in caption_phrases_bboxs_label:
        if label not in unique_phrases_bboxs_label:
            unique_phrases_bboxs_label.append(label)

    # print('unique_phrases_bboxs_label: ', unique_phrases_bboxs_label)
    for label in unique_phrases_bboxs_label:
        label_index = [idx for idx in range(len(caption_phrases_bboxs_label))
                       if label == caption_phrases_bboxs_label[idx]]
        phrases_bboxs.append(caption_phrases_bboxs[label_index].tolist())
        phrases_bboxs_label.append(label)
        phrases_cate.append(caption_phrases_cate[label_index][:, 0].tolist()[0])

    return phrases_bboxs, phrases_bboxs_label, phrases_cate


def is_invaild_bboxs(bbox_coordinates):
    is_invaild = list()
    for bbox_coor in bbox_coordinates:
        if -1 in bbox_coor:
            is_invaild.append(bbox_coor)
    return len(is_invaild) == len(bbox_coordinates)



def overlap_fp_ROIS_stata(mapped_ROIS, iou_scores, state_type="value", is_dis_weighted=False):
    ''' Calculating the matching scores of each position in the feature map
        
        Args:
            mapped_ROIS: Numpy array with shape [num_anchors, 4], 
                        coordinate is [upper_y, upper_x, bottom_y, bottom_x],
                        each element i, j in it correponds i, j in iters_iou_scores
            iou_scores: Numpy array with shape [num_phrases, num_anchors]

        Return:
            pos_values: Numpy array with shape [num_phrases, feature_map_h, feature_map_w]
    '''
    fp_shape = mapped_ROIS[-1, 2:]

    fp_start = mapped_ROIS[0, :2]
    st_h = fp_start[0]
    st_w = fp_start[1]
    end_h = fp_shape[0]
    end_w = fp_shape[1]

    # 1. initial each pos and its value
    num_phrases = len(iou_scores)
    pos_values = np.zeros((num_phrases, int(end_h+1), int(end_w+1)), dtype=float)

    # search all the mapped ROIs and assign the value
    for ph_idx in range(num_phrases):
        for mpd_idx in range(len(mapped_ROIS)):
            mapped_ROI_coor = mapped_ROIS[mpd_idx, :]
            ROI_value = iou_scores[ph_idx, mpd_idx]
            [ROI_ymin, ROI_xmin, ROI_ymax, ROI_xmax] = [mapped_ROI_coor[0], mapped_ROI_coor[1],
                                                        mapped_ROI_coor[2], mapped_ROI_coor[3]]

            if is_dis_weighted:
                ROI_h = ROI_ymax - ROI_ymin
                ROI_w = ROI_xmax - ROI_xmin
                center_pos = np.array([ROI_ymin + int(ROI_h/2),
                                        ROI_xmin + int(ROI_w/2)])

            for cur_st_y in range(int(ROI_ymin), int(ROI_ymax)+1):
                for cur_st_x in range(int(ROI_xmin), int(ROI_xmax)+1):
                    cur_pos = np.array([cur_st_y-st_h, cur_st_x-st_w], dtype=int)

                    if is_dis_weighted:
                        dis = np.sum(np.square(center_pos - cur_pos))

                        dis_weight = 1 / np.exp(dis) 

                    else:
                        dis_weight = 1

                    if state_type is "value":
                        pos_values[ph_idx, cur_pos[0], cur_pos[1]] += dis_weight * ROI_value

                    elif state_type is "count":
                        pos_values[ph_idx, cur_pos[0], cur_pos[1]] += dis_weight * 1

    return pos_values