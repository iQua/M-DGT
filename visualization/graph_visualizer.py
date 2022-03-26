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
import scipy as scip
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


''' custom libs '''
from .Configs import mat_colors_digial_map, mat_digital_colors_map
# from .Configs import colors_digial_map, digital_colors_map, colors_cmp_map
from .utils import numbering_directory, phrases_to_file

class GraphVisualizer(object):
    '''Define a visualizer for the graph.
  
    Attributes:
        visualization_dir: the directory that saves the results.
    
    Example:

'''

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


    def set_base_save_dir(self, base_save_dir):
        self._vis_log_dir = base_save_dir

    def set_save_dir(self, epoch_number, step_i):
        self._vis_log_dir = os.path.join(self._vis_log_dir, str(epoch_number) + "_" +str(step_i))
        os.makedirs(self._vis_log_dir, exist_ok=True)

    def reset_save_dir(self):
        self._vis_log_dir = self.reset_holder 


    def visualize_graph(self, graph, save_path, baseline_center, color='red'):

        x_max = baseline_center[0]
        y_max = baseline_center[1]
        
        nodes_color = None
        nodes_label = graph.nodes.get_nodes_label()
        nodes_color = [mat_digital_colors_map[int(n_label)] for n_label in nodes_label]

        visual_graph = graph.spatial_graph

        nodes_pos = nx.get_node_attributes(visual_graph,'pos')
        # print("x_max: ", x_max)
        # print("y_max: ", y_max)
        # print("nodes_pos: ", nodes_pos)

        required_nodes_list = list(nodes_pos.keys())
        # print("nodes_pos: ", nodes_pos)
        # print("required_nodes_list: ", required_nodes_list)
        # print("visual_graph.edges(): ", visual_graph.edges())

        nodes_size_info = nx.get_node_attributes(visual_graph,'score') # {node_id: node_score}
        nodes_size = [nodes_size_info[node_id] for node_id in list(nodes_size_info.keys())]
 
        fig, ax = plt.subplots()
        nx.draw(visual_graph, pos = nodes_pos, ax=ax,
                nodelist = required_nodes_list, 
                node_size = nodes_size,
                node_color = nodes_color, # node_color=color_map
                width=2, alpha=0.5,
                #cmap = cmap,
                with_labels=True,
                )

        # norm = mpl.colors.Normalize(vmin=0, vmax=1)

        # cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
        #                                 norm=norm,
        #                                 orientation='horizontal')
        # cb1.set_label('Matching Scores')
        limits=plt.axis('on') # turns on axis
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        ax.set_xlim(0, x_max+10)
        ax.set_ylim(0, y_max+10)

        ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
        ax.xaxis.tick_top()                     # and move the X-Axis      
        ax.yaxis.tick_left()   
        plt.axis('off')
        plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
        plt.close()
        # print(ok)

    def graphs_visualization(self, built_physical_graphs, pre_save_name, log_name="sample_graphs"):
        """[Visualize the built physical graphs]

        Args:
            built_physical_graphs (OrderDict(str: graph)): [an order that contains the built graphs]
        """

        base_imgs_boxes_save_path = os.path.join(self._vis_log_dir, log_name)

        os.makedirs(base_imgs_boxes_save_path, exist_ok=True)

        graphs_id = built_physical_graphs.keys()

        initial_nodes_center = np.array(built_physical_graphs[0].nodes.get_nodes_center())
        baselin_center_max = np.amax(initial_nodes_center, axis=0)

        for graph_id in list(graphs_id):
            # print("\n")
            # print("graph_id: ", graph_id)
            
            required_range = baselin_center_max

            built_phy_graph = built_physical_graphs[graph_id]
            cur_nodes_center = np.array(built_phy_graph.nodes.get_nodes_center())
            cur_center_max = np.amax(cur_nodes_center, axis=0)

            if cur_center_max[0] > required_range[0]:
                required_range[0] = cur_center_max[0]

            if cur_center_max[1] > required_range[1]:
                required_range[1] = cur_center_max[1]

            save_path = os.path.join(base_imgs_boxes_save_path, "graph_" + str(graph_id) + pre_save_name + '_SpatialGrpah.png')


            self.visualize_graph(graph=built_phy_graph, save_path=save_path, color='red', baseline_center=required_range)



