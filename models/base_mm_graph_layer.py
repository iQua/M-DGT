#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import scipy as scip
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from models.base_graph_op import Nodes
from models.node_transformer import generate_coord, generate_relative_coor


# The GATLayer is supported by a public repo "https://github.com/gordicaleksa/pytorch-GAT/blob/main/models/definitions/GAT.py"
class CMGATLayerBase(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.
    """

    head_dim = 1

    def __init__(self,
                 cross_modal_attn_model,
                 num_in_node_features,
                 num_in_phrase_features,
                 num_out_node_features,
                 num_out_phrase_features,
                 num_of_heads,
                 concat=True,
                 activation=nn.ELU(),
                 dropout_prob=0.6,
                 add_skip_connection=True,
                 bias=True,
                 log_attention_weights=False):

        super().__init__()

        # Saving the CrossModalContextMultiHeadedAttention for furhter utilization
        self.cm_attn_model = cross_modal_attn_model

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_node_features = num_out_node_features
        self.num_out_phrase_features = num_out_phrase_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights: linear projection matrix attention target/source
        # (denoted as "a" in the paper) and bias
        #
        # You can treat this one matrix as num_of_heads independent matrices
        self.node_linear_proj = nn.Linear(num_in_node_features,
                                          num_of_heads * num_out_node_features,
                                          bias=False)

        self.node_spatial_linear_proj = nn.Linear(8, 256, bias=False)

        self.edge_relative_spatial_linear_proj = nn.Linear(4, 256, bias=False)

        self.phrase_linear_proj = nn.Linear(num_in_phrase_features,
                                            num_of_heads *
                                            num_out_phrase_features,
                                            bias=False)
        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(
                torch.Tensor(num_of_heads * num_out_node_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_node_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_node_features,
                                       num_of_heads * num_out_node_features,
                                       bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(
            0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(
            dim=-1
        )  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
        """
        nn.init.xavier_uniform_(self.node_linear_proj.weight)
        nn.init.xavier_uniform_(self.node_spatial_linear_proj.weight)
        nn.init.xavier_uniform_(self.edge_relative_spatial_linear_proj.weight)
        nn.init.xavier_uniform_(self.phrase_linear_proj.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def obtain_node_spatial_info(self, node_boxes, base_image_Hs,
                                 base_image_Ws):
        node_spatial_coor = generate_coord(node_boxes, base_image_Hs,
                                           base_image_Ws)
        return node_spatial_coor

    def obtain_nodes_relative_spatial_info(self, source_nodes_boxes,
                                           target_nodes_boxes):
        nodes_relative_coors = generate_relative_coor(source_nodes_boxes,
                                                      target_nodes_boxes)
        return nodes_relative_coors

    def skip_concat_bias(self, attention_coefficients, in_nodes_features,
                         out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[
                    -1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(
                    -1, self.num_of_heads, self.num_out_node_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(
                -1, self.num_of_heads * self.num_out_node_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(
            out_nodes_features)
