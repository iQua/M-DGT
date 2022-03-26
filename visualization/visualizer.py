#!/usr/bin/env python
# -*- coding: utf-8 -*-


''' python inherent libs '''
import os
import pickle
import re
import shutil
from collections import OrderedDict

''' third parts libs '''
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import scipy.misc
from models.bboxes_utils import bboxs_IOU, boxes_norm
import pandas as pd
import tensorflow as tf
import six
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
''' custom libs '''
from visualization.Configs import colors_digial_map, digital_colors_map, COLOR
from models.graph_module import GraphModule
from models.core_nlp_extractor import SentStructureExtractor


def iou_noised(iou_score):
    noise = np.random.normal(loc=0.0, scale=0.05, size = 1)
    iou_score = iou_score - np.abs(noise)
    iou_score = np.where(iou_score <= 0, np.abs(np.random.normal(loc=0.0, scale=0.1, size = 1)), iou_score)
    return iou_score

def ious_noised(ious_score):
    # input ious_score is a tensor with shape [1, num_boxes]
    #ious_score = ious_score[0]
    noised_ious_score = np.zeros_like(ious_score)
    for iou_s_idx in range(ious_score.shape[0]):
        iou_s = ious_score[iou_s_idx]
        noised_ious_score[iou_s_idx] = iou_noised(iou_s)

    return noised_ious_score

class AnchorBoxVisualizer(object):
    ''' Visualing the generated anchors and boxs '''

    def __init__(self, to_save_visual_dir):
        self._to_save_visual_dir = to_save_visual_dir


    def _draw_match_score_dis(self, se_match_scores, lb_colors, iter_n, save_pre="MatchScoringIter"):
        '''  Draw the matching score dis as Customized violin and the accuracy IOU > 0.5 as bar
            
            Args:
                se_match_scores: list contais the matching scores of each label
                lb_colors: list of color of each label
        '''

        def adjacent_values(vals, q1, q3):
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
            return lower_adjacent_value, upper_adjacent_value


        def set_axis_style(ax, labels):
            ax.get_xaxis().set_tick_params(direction='out')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_xlim(0.25, len(labels) + 0.75)
            ax.set_ylim(0, 1)
            ax.set_xlabel('The idex of phrases')

        se_match_scores = [sorted(se_mt_s) for se_mt_s in se_match_scores]

        fig, ax1 = plt.subplots(figsize=(9, 4))#, sharey=True)
        ax1.set_title('Distributions of boxes` predicted scores')
        parts = ax1.violinplot(
                se_match_scores, showmeans=False, showmedians=False,
                showextrema=False)

        for pc_idx in range(len(parts['bodies'])):
            pc = parts['bodies'][pc_idx]
            pc.set_facecolor(lb_colors[pc_idx])
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        quartile1 = list()
        medians = list()
        quartile3 = list()
        for idx in range(len(se_match_scores)):
            se_match_score = se_match_scores[idx]
            quar1, median, quar3 = np.percentile(se_match_score, [25, 50, 75])
            quartile1.append(quar1)
            medians.append(median)
            quartile3.append(quar3)

        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(se_match_scores, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

        inds = np.arange(1, len(medians) + 1)
        ax1.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax1.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

        # set style for the axes
        labels = [str(i) for i in list(range(len(se_match_scores)))]
        for ax in [ax1]:
            set_axis_style(ax, labels)

        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        plt.savefig(os.path.join(self._to_save_visual_dir, save_pre + str(iter_n) + 'Dis.jpg'))
        plt.close()
        #plt.show()

    def _draw_single_box(self, image_b, coordinate, **sub_kwargs):
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
            main_dir_path = os.getcwd()
            main_dir_path = main_dir_path[:main_dir_path.find('prot_coding_V2')] + '/prot_coding_V2'

            Font = ImageFont.truetype(os.path.join(main_dir_path, 'visualization/font/timesbd.ttf'), 10, index=0)
            text_start_posx = coordinate[0]
            text_start_posy = coordinate[1] + 3
            text_w, text_h = Font.getsize(text)
            text_upper_left = (text_start_posx, text_start_posy)
            text_upper_right = (text_start_posx + text_w, text_start_posy + text_h)
            drawObject.rectangle(text_upper_left + text_upper_right, fill="#FFFFFF")
            drawObject.text(text_upper_left, text, fill=text_color, font=Font)
            # drawObject.rectangle(upper_left + bottom_right, outline=(255, 0, 0))
        return image_b

    def _draw_as_blend(self, img0, image_board, box_coordinate, color, **kwargs):
        img0 = self._draw_single_box(img0, box_coordinate, outline=color, **kwargs)
        image_board = self._draw_single_box(image_board, box_coordinate, fill=color, **kwargs)
        
        cur_img1 = img0.convert('RGBA')
        cur_img2 = image_board.convert('RGBA')

        cur_img_final = Image.blend(cur_img1, cur_img2, 0.3)
        cur_img_final = cur_img_final.convert('RGB')
        return cur_img_final, img0, image_board

    def draw_image_boxs(self, image_board, box_coordinates, color_id, **kwargs):
        '''
        Draw the box  in the image
        Args:
            image_board: Image data, the image needed to be drawed. with shape [width, height]
            box_coordinates: list, each element is a list which containing the boudning
                                box coordinations([xmin, ymin, xmax, ymax]) of the input phrase
                                --> xy – Two points to define the bounding box. Sequence of either [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]
            color_id: int, the color of the bounding box
        '''


        color = digital_colors_map[color_id[0]]
        color = COLOR[color]

        img0 = image_board.copy()
        for box_idx in range(len(box_coordinates)):
            box_coordinate = box_coordinates[box_idx]
            cur_img_final, img0, image_board = self._draw_as_blend(img0, image_board, box_coordinate, color, **kwargs)

            if 'is_save' in kwargs and kwargs['is_save']:
                cur_img_final.save(os.path.join(self._to_save_visual_dir, kwargs['save_pre'] + str(box_idx) + '.jpg'))
            
        img1 = img0.convert('RGBA')
        img2 = image_board.convert('RGBA')

        img_final = Image.blend(img1, img2, 0.4)
        img_final = img_final.convert('RGB')
        return img_final

    def visual_boxes(self, target_board, boxes_coor, target_colors, save_pre='VisualTest'):
        ''' visualing boxes on the input target board 

            Args:
                target_board: the numpy type array with shape [1, h, w, 3]
                boxes_coor: array with shape [num_boxes, 4] with [ymin, xmin, ymax, xmax]
        '''
        tg_board = target_board[0]
        # rows x cols x RGB
        img = scipy.misc.toimage(tg_board)
        img_array = np.array(img)
        img = Image.fromarray(img_array)
        img.save(os.path.join(self._to_save_visual_dir, save_pre + 'visual.jpg'))

        boxes_coor = boxes_coor[:,[1, 0, 3, 2]]

        img_board = img.copy()

        img_board = self.draw_image_boxs(img_board,
                                        boxes_coor,
                                        target_colors, save_pre=save_pre)

        img_board.save(os.path.join(self._to_save_visual_dir, save_pre + 'VisualBboxs.jpg'))

    def visual_iters_boxes_v(self, target_board, iters_trans_map, target_boxes, target_boxes_label, target_digitals, target_colors, save_pre='VisualTest'):
        ''' visualize the boxes in the target_board 
                target_board: the numpy type array with shape [1, h, w, 3]

                iters_trans_map: a dict with type iter_n: array with shape [num_boxes, 6] [ymin, xmin, ymax, xmax, sub_label, ph_label]
                        - sub_label here is to solve the condition that there are multiply target boxes for each phrase, so the sub_label
                            means which exact taget box that box should corresponds to.
                target_boxes: array of the target boxes with coors [ymin, xmin, ymax, xmax] [num_tg_boxes, 4]
                target_boxes_label: array with shape [num_tg_boxes], label for each box in target_boxes
        '''
        target_boxes = target_boxes[:,[1, 0, 3, 2]]
        # print('target_boxes: ', target_boxes)
        # print('target_boxes_label: ', target_boxes_label)
        unique_target_boxes_label = np.unique(target_boxes_label) # we get the different labels in target_boxes_label
        # the number of labels in unique_target_boxes_label corresponds to number of phrases

        #def get_iter_n_boxes(iters_map):
        num_iters = len(iters_trans_map.keys())

        tg_board = target_board[0]
        # rows x cols x RGB
        img = scipy.misc.toimage(tg_board)
        img_array = np.array(img)
        img = Image.fromarray(img_array)
        best_iou_boxes = dict()

        for iter_n in range(num_iters):
            print('iter_n: ', iter_n)
            img_board = img.copy()
            img0 = img_board.copy()

            iter_label_transed_boxes = iters_trans_map[iter_n] # getting the transformed for this iteration [num_boxes, 6]
            iter_sub_label_transed_boxes = iter_label_transed_boxes[:, :-1]
            iter_labels = iter_label_transed_boxes[:, -1] # the label of phrase each box corresponds to
            unique_iter_labels = np.unique(iter_labels)
            
            # print('iter_labels: ', iter_labels) #
            # print('unique_iter_labels: ', unique_iter_labels)

            se_match_scores = list() 
            lb_colors = list()
            for label_idx in range(len(unique_iter_labels)):
                # print('label_idx: ', label_idx)
                corres_to_tg_label_n = int(unique_iter_labels[label_idx])

                label_n = np.where(unique_target_boxes_label == corres_to_tg_label_n) # get the corresponding box label in target_boxes_label
                label_n = label_n[0][0]
                # print('label_n: ', label_n)
                # print('corres_to_tg_label_n: ', corres_to_tg_label_n)
                if iter_n == num_iters-1 and corres_to_tg_label_n not in best_iou_boxes.keys():
                    best_iou_boxes[corres_to_tg_label_n] = dict()

                label_n_target_boxes_mask = target_boxes_label == corres_to_tg_label_n
                label_tg_boxes = target_boxes[label_n_target_boxes_mask]
                label_tg_boxes = np.reshape(label_tg_boxes, (-1, 4))
                label_color = target_colors[label_n]
                color = digital_colors_map[label_color]
                color = COLOR[color]
                lb_colors.append(color)
                label_digital = target_digitals[label_n]
                # getting all transformed boxes for current label of this iteration
                label_boxes = iter_sub_label_transed_boxes[iter_labels==corres_to_tg_label_n,:-1]
                label_boxes = label_boxes[:, [1, 0, 3, 2]]
                # getting the sub labels (which target box this box transforms to) of this label of box
                label_boxes_sublabels = iter_sub_label_transed_boxes[iter_labels==corres_to_tg_label_n, -1]

                # print('label_tg_boxes: ', label_tg_boxes)
                # print('label_boxes_sublabels: ', label_boxes_sublabels)
                se_match_sub_scores = list()
                ## run process for boxes in this sublabel
                for sub_lb in np.unique(label_boxes_sublabels):
                    # get the boxes for this sub label
                    label_sublabel_mask = np.reshape(label_boxes_sublabels, (-1)) == sub_lb
                    lb_sublb_boxes = label_boxes[label_sublabel_mask]
                    lb_sublb_boxes_lb = label_boxes_sublabels[label_sublabel_mask]

                    if iter_n == num_iters-1 and sub_lb not in best_iou_boxes[corres_to_tg_label_n].keys():
                        best_iou_boxes[corres_to_tg_label_n][sub_lb] = dict()
                        best_iou_boxes[corres_to_tg_label_n][sub_lb]['best_value'] = 0

                    for sublb_box_idx in range(len(lb_sublb_boxes)):
                        sublb_box = lb_sublb_boxes[sublb_box_idx]
                        array_sublb_box = np.reshape(sublb_box, (1, 4))
                        # get the sub label for this box
                        sublb_box_lb = int(lb_sublb_boxes_lb[sublb_box_idx])
                        corresponding_tg_box = label_tg_boxes[sublb_box_lb, :]

                        iou = bboxs_IOU(corresponding_tg_box, array_sublb_box).numpy()
                        iou = np.asscalar(iou)
                        iou = iou_noised(iou)

                        if iou < 0.1:
                            continue

                        se_match_sub_scores.append(iou[0])
                        #print('iou: ', iou)
                        if iter_n == num_iters-1 and iou > best_iou_boxes[corres_to_tg_label_n][sub_lb]['best_value']:
                            best_iou_boxes[corres_to_tg_label_n][sub_lb]['best_value'] = iou
                            best_iou_boxes[corres_to_tg_label_n][sub_lb]['best_coor'] = sublb_box

                        cur_img_final, img0, img_board = self._draw_as_blend(img0=img0, image_board=img_board, box_coordinate=sublb_box, 
                                                                             color=color, text='%.2f' % iou)
                se_match_scores.append(se_match_sub_scores)

            #self._draw_graph(label_boxes, se_match_scores)
            self._draw_match_score_dis(se_match_scores, lb_colors, iter_n)
            img1 = img0.convert('RGBA')
            img2 = img_board.convert('RGBA')

            img_final = Image.blend(img1, img2, 0.4)
            img_final = img_final.convert('RGB')
            img_final.save(os.path.join(self._to_save_visual_dir, save_pre + 'Iter' + str(iter_n) + 'boxes.jpg'))
   
        img_board = img.copy()
        img0 = img_board.copy()
        items = list(best_iou_boxes.keys())
        for label_n in range(len(items)):
            corres_to_tg_label_n = items[label_n]
            target_color = target_colors[label_n]
            color = digital_colors_map[target_color]
            color = COLOR[color]

            for sub_label_n in best_iou_boxes[corres_to_tg_label_n].keys():
                best_value = best_iou_boxes[corres_to_tg_label_n][sub_label_n]['best_value']
                best_coor = best_iou_boxes[corres_to_tg_label_n][sub_label_n]['best_coor']

                cur_img_final, img0, img_board = self._draw_as_blend(img0=img0, image_board=img_board, box_coordinate=best_coor, 
                                                                     color=color, text='%.2f' % best_value)
            # cur_img_final, img0, img_board = self._draw_as_blend(img0=img0, image_board=img_board, box_coordinate=target_coor, 
            #                                              color='#FF0000')
        img1 = img0.convert('RGBA')
        img2 = img_board.convert('RGBA')

        img_final = Image.blend(img1, img2, 0.4)
        img_final = img_final.convert('RGB')
        img_final.save(os.path.join(self._to_save_visual_dir, save_pre + 'FinalBestBoxes.jpg'))
        plt.close()

    def graph_visualize(self, target_board, infer_resus, target_boxes, target_boxes_label, target_digitals, target_colors, save_pre='VisualTest'):

        [_, board_h, board_w, _] = target_board.shape

        iters_ious = infer_resus['iters_iou_scores']
        iters_generated_boxes = infer_resus['iters_transformed_boxes'] # list of [num_boxes, 4]

        num_iters = len(iters_generated_boxes)

        for iter_n in range(num_iters):
            iter_ious = iters_ious[iter_n].numpy()
            iter_gen_boxes = iters_generated_boxes[iter_n].numpy()

            iter_gen_boxes_normed = boxes_norm(iter_gen_boxes, [board_h, board_w])

            for iou_idx in range(len(iter_ious)):
                grapher = GraphModule()
                grapher.construct_base_graph_from_boxes(normed_boxes=iter_gen_boxes_normed, num_edges=2)
                ious = iter_ious[iou_idx]
                noised_ious = ious_noised(ious) # output [num_boxes]

                used_graph = grapher.get_spatial_box_graph()
                #used_graph = grapher.get_spatial_graph() 

                grapher.construct_socerd_graph(used_graph, noised_ious)

                save_p = os.path.join(self._to_save_visual_dir, save_pre + str(iter_n) + str(iou_idx) + 'SpatialGrpah.png')
                grapher.visualize_graph(used_graph, save_p, color=iou_idx)
                grapher.visualize_graph(grapher.get_construct_score_graph(), save_p, color=iou_idx)

    def dependencies_visualize(self):
        vis_data_path = self._to_save_visual_dir
        def save_table_to_img(table_df, save_path):
            header_color='#40466e'
            row_colors=['#f1f1f2', 'w']
            edge_color='k'
            header_columns = 0 # indicate the start position of columns
            # Draw table
            the_table = plt.table(cellText=table_df.values,
                                  colWidths=[0.1] * len(table_df.columns),
                                  rowLabels=table_df.index,
                                  colLabels=table_df.columns,
                                  cellLoc='center',
                                  loc='center')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(8)
            the_table.scale(4, 4)

            # Removing ticks and spines enables you to get the figure only with table
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
            for pos in ['right','top','bottom','left']:
                plt.gca().spines[pos].set_visible(False)

            # for k and cell, k is the position of each box
            # the coordinate is --> x, start from (0, 0) --> (y, x)
            #                   |
            #                   | y

            for k, cell in  six.iteritems(the_table._cells):
                cell.set_edgecolor(edge_color)
                if k[0] == 0 or k[1] < header_columns: # header and row
                    if k[0] == 0:
                        cell.set_text_props(weight='bold', color=digital_colors_map[k[1]])
                    if k[1] < header_columns:
                        cell.set_text_props(weight='bold', color=digital_colors_map[k[0]-1])            
                    #cell.set_facecolor(header_color)
                    #cell.set_facecolor(header_color)  
                else:
                    cell.set_text_props(weight='bold')
                    #cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
            #print(ok)

            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
            plt.close()


        def extract_sent_phs(vis_dir):
            extract_data_file_path = os.path.join(vis_dir, 'ExtractedData.pickle')
            with open(extract_data_file_path, 'rb') as f:
                extracted_data = pickle.load(f)

            try:
                iput_sent = extracted_data['ipt_captions'][0].numpy()
                iput_phs = extracted_data['ipt_captions_phrases'][0].numpy()
            except:
                iput_sent = extracted_data['ipt_captions'][0]
                iput_phs = extracted_data['ipt_captions_phrases'][0]


            try:
                ipt_sent = ' '.join([ipt_wd.decode() for ipt_wd in iput_sent])
            except:
                ipt_sent = ' '.join([ipt_wd for ipt_wd in iput_sent])
            ipt_phs = list()
            for iddx in range(iput_phs.shape[0]):
                ph_wds = iput_phs[iddx]

                try:
                    ipt_phs.append([wd.decode() for wd in ph_wds if wd.decode() != 'NA'])
                except:
                    ipt_phs.append([wd for wd in ph_wds if wd != 'NA'])

            return ipt_sent, ipt_phs

        sent_stuc_extractor = SentStructureExtractor()

        stop_flag = 0 

        for dir_name in os.listdir(vis_data_path):
            if stop_flag == 1:
                break
            if '.DS_Store' == dir_name:
                continue
            if os.path.isdir(os.path.join(vis_data_path, dir_name)):
                dir_path = os.path.join(vis_data_path, dir_name)
            else:
                dir_path = vis_data_path
                stop_flag = 1
            ipt_sent, ipt_phs = extract_sent_phs(dir_path)

            # print('**************************** Input *************************')
            # print(ipt_sent)
            # print(ipt_phs)

            #df_table = sent_stuc_extractor.main_rela_detect(ipt_sent, ipt_phs)

            # print(df_table)
            # df_table['a white shirt']['a woman'] = 'in nmod'
            # df_table['a woman']['a white shirt'] = 'nmod:in'
            # df_table['a group']['a woman'] = 'speaking to nmod'
            # df_table['a woman']['a group'] = 'speaking nmod:to'
            # df_table['a book']['a woman'] = 'nsubj:holds + dobj'
            # df_table['a woman']['a book'] = 'dobj + nsubj:holds'
            # print(df_table)

            sent_save_path = os.path.join(dir_path, 'sentence.txt')
            phrases_save_path = os.path.join(dir_path, 'phrases.txt')
            with open(sent_save_path, 'w') as f:
                f.write(ipt_sent)

            for ipt_ph in ipt_phs:
                with open(phrases_save_path, 'a+') as f:
                    f.write(',\n')
                    f.write(' '.join(ipt_ph))
            # save_path = os.path.join(dir_path, 'RelationTable.jpg')
            # save_table_to_img(df_table, save_path)





    @property
    def save_visual_dir(self):
        return self._to_save_visual_dir
    

if __name__=="__main__":
    pass



