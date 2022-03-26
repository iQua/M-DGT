#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Inherent Python '''
import os
import pickle
import re
import shutil
from collections import OrderedDict


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
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
''' custom libs '''
from visualization.Configs import colors_digial_map, digital_colors_map, COLOR



class StatisticalVisualizer(object):
    """ Visualizing the statistical measurements"""


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