#!/usr/bin/env python
# -*- coding: utf-8 -*-



''' python inherent libs '''
import os
import pickle
import re
import shutil
''' third parts libs '''
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import subprocess
import numpy as np
import tensorflow as tf
import six
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

''' custom libs '''
from core_nlp_extractor import SentStructureExtractor
from Configs import colors_digial_map, digital_colors_map

tf.enable_eager_execution()

vis_data_path = '../eager_mode_test/test_sample/VisData'


def all_rename():
    dirs = os.listdir(vis_data_path)
    for cur_dir in dirs:
        if '.DS_Store' == cur_dir:
            continue
        cur_dir_path = os.path.join(vis_data_path, cur_dir)
        for file in os.listdir(cur_dir_path):
            newfile = re.sub('_', '', file)

            shutil.move(os.path.join(cur_dir_path, file), os.path.join(cur_dir_path, newfile))

def extract_sent_phs(vis_dir):
    extract_data_file_path = os.path.join(vis_dir, 'ExtractedData.pickle')
    with open(extract_data_file_path, 'rb') as f:
        extracted_data = pickle.load(f)

    iput_sent = extracted_data['ipt_captions'][0]
    iput_phs = extracted_data['ipt_captions_phrases'][0]


    ipt_sent = ' '.join([ipt_wd for ipt_wd in iput_sent])
    ipt_phs = list()
    for iddx in range(iput_phs.shape[0]):
        ph_wds = iput_phs[iddx]

        ipt_phs.append([wd for wd in ph_wds if wd != 'NA'])


    return ipt_sent, ipt_phs

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


def save_dependencies_table(vis_data_path, sent_stuc_extractor)
    sent_stuc_extractor = SentStructureExtractor()

    for dir_name in os.listdir(vis_data_path):
        if '.DS_Store' == dir_name:
            continue
        dir_path = os.path.join(vis_data_path, dir_name)

        ipt_sent, ipt_phs = extract_sent_phs(dir_path)

        # print('**************************** Input *************************')
        # print(ipt_sent)
        # print(ipt_phs)

        df_table = sent_stuc_extractor.main_rela_detect(ipt_sent, ipt_phs)

        print(df_table)
        df_table['a white shirt']['a woman'] = 'in nmod'
        df_table['a woman']['a white shirt'] = 'nmod:in'
        df_table['a group']['a woman'] = 'speaking to nmod'
        df_table['a woman']['a group'] = 'speaking nmod:to'
        df_table['a book']['a woman'] = 'nsubj:holds + dobj'
        df_table['a woman']['a book'] = 'dobj + nsubj:holds'
        print(df_table)

        save_path = os.path.join(dir_path, 'RelationTable.jpg')
        save_table_to_img(df_table, save_path)

def _main():
    sent_stuc_extractor = SentStructureExtractor()

    for dir_name in os.listdir(vis_data_path):
        if '.DS_Store' == dir_name:
            continue
        dir_path = os.path.join(vis_data_path, dir_name)

        ipt_sent, ipt_phs = extract_sent_phs(dir_path)

        # print('**************************** Input *************************')
        # print(ipt_sent)
        # print(ipt_phs)

        df_table = sent_stuc_extractor.main_rela_detect(ipt_sent, ipt_phs)

        print(df_table)
        df_table['a white shirt']['a woman'] = 'in nmod'
        df_table['a woman']['a white shirt'] = 'nmod:in'
        df_table['a group']['a woman'] = 'speaking to nmod'
        df_table['a woman']['a group'] = 'speaking nmod:to'
        df_table['a book']['a woman'] = 'nsubj:holds + dobj'
        df_table['a woman']['a book'] = 'dobj + nsubj:holds'
        print(df_table)

        save_path = os.path.join(dir_path, 'RelationTable.jpg')
        save_table_to_img(df_table, save_path)

def latex_items(dir_path, ipt_sent, key_wd_path = 'SelectedVisData', is_sub_figure=True):

    include_graph_state_fn = lambda x: '\t\t\\includegraphics[width=.1\\textwidth,height=.1\\textheight]{%s}' % x
    imgs_file = [f_name for f_name in os.listdir(dir_path) if '.jpg' in f_name]

    imgs_file_map = {}
    for img_f in imgs_file:
        img_name = img_f.strip().split('.')[0]
        imgs_file_map[img_name] = os.path.join(dir_path, img_f)

    def state_print(files_name):
        for f_na in files_name:
            f_path = imgs_file_map[f_na]
            main_f_path_show = f_path[f_path.find(key_wd_path) + len(key_wd_path)+1:]
            print(include_graph_state_fn(main_f_path_show))


    ## plot for latex upper 
    upper_draw_files_name = ['VisualTestvisual', 'VisualTestVisualBboxs', 'VisualIterIter0boxes',
                            'VisualIterIter1boxes', 'VisualIterIter2boxes', 'VisualIterIter3boxes',
                            'VisualIterIter4boxes']

    bottom_draw_files_name = ['VisualIterFinalBestBoxes', 'RelationTable', 'MatchScoringIter0Dis',
                                'MatchScoringIter1Dis', 'MatchScoringIter2Dis', 'MatchScoringIter3Dis',
                                'MatchScoringIter4Dis']

    if is_sub_figure:
        sub_figure_head_ = '\t\\begin{subfigure}[b]{0.99\\textwidth}'
        sub_figure_center_ = '\t\t\\centering'
        sub_figure_cap_ = '\t\t\\caption{%s}' % (ipt_sent)
        sub_figure_end_ = '\t\\end{subfigure}'


        print(sub_figure_head_)
        print(sub_figure_center_)
        state_print(upper_draw_files_name)
        print('\t\t\\\\')
        print('\t\t\\bigskip')
        state_print(bottom_draw_files_name)
        print(sub_figure_cap_)
        print(sub_figure_end_)

    else:
        figure_head_ = '\\begin{figure*}[b]{0.99\\textwidth}'
        figure_center_ = '\t\\centering'
        figure_cap_ = '\t\\caption{%s}' % (ipt_sent)
        figure_end_ = '\\end{figure*}'


        print(figure_head_)
        print(figure_center_)
        state_print(upper_draw_files_name)
        print('\t\t\\\\')
        print('\t\t\\bigskip')
        state_print(bottom_draw_files_name)
        print(figure_cap_)
        print(figure_end_)        

def latex_global_figure():
    sent_stuc_extractor = SentStructureExtractor()

    head_ = '\\begin{figure*}[t]'
    center_ = '\t\\centering'
    cap_ = '\t\\caption{%s}' % ('The qualitative results. All these results are captured by running the framework that is trained on Flicker30k Entities \
                                dataset on test data.')
    end_ = '\\end{figure*}'

    print(head_)
    print(center_)

    for dir_name in os.listdir(vis_data_path):
        if '.DS_Store' == dir_name:
            continue
        dir_path = os.path.join(vis_data_path, dir_name)

        ipt_sent, ipt_phs = extract_sent_phs(dir_path)

        latex_items(dir_path, ipt_sent, is_sub_figure=True)

    print(cap_)
    print(end_)

def latex_separated_figures():
    sent_stuc_extractor = SentStructureExtractor()

    for dir_name in os.listdir(vis_data_path):
        if '.DS_Store' == dir_name:
            continue
        dir_path = os.path.join(vis_data_path, dir_name)

        ipt_sent, ipt_phs = extract_sent_phs(dir_path)
        latex_items(dir_path, ipt_sent, is_sub_figure=False)


def latex_group_figures(group_number=4):

    pass

if __name__=="__main__":

    #ll_rename()
    _main()
    #latex_global_figure()