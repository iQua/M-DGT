#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from .utils import clones


# this part of code is obtained from the paper "A Fast and Accurate One-Stage Approach to Visual Grounding"
def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    xv, yv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
    xv_min = (xv.float() * 2 - width) / width
    yv_min = (yv.float() * 2 - height) / height
    xv_max = ((xv + 1).float() * 2 - width) / width
    yv_max = ((yv + 1).float() * 2 - height) / height
    xv_ctr = (xv_min + xv_max) / 2
    yv_ctr = (yv_min + yv_max) / 2
    hmap = torch.ones(height, width) * (1. / height)
    wmap = torch.ones(height, width) * (1. / width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch, 1, 1, 1)
    return coord


def generate_relative_coor(source_boxes, target_boxes):
    """ 
        Compute the relative spatial coor for these two sets of boxes. 
        source_boxes and target_boxes should be in the same shape. Each row of them corresponds to each other
    
    """
    ws_i = torch.index_select(source_boxes, 1, 2) - torch.index_select(
        source_boxes, 1, 0)
    hs_i = torch.index_select(source_boxes, 1, 3) - torch.index_select(
        source_boxes, 1, 1)

    ws_j = torch.index_select(target_boxes, 1, 2) - torch.index_select(
        target_boxes, 1, 0)
    hs_j = torch.index_select(target_boxes, 1, 3) - torch.index_select(
        target_boxes, 1, 1)
    # x_imin - x_jmin
    xs_i_min = torch.index_select(source_boxes, 1, 0)
    xs_j_min = torch.index_select(target_boxes, 1, 0)
    xs_ij_delta = torch.abs(xs_i_min - xs_j_min) / ws_i
    # y_imin - y_jmin
    ys_i_min = torch.index_select(source_boxes, 1, 2)
    ys_j_min = torch.index_select(target_boxes, 1, 2)
    ys_ij_delta = torch.abs(ys_i_min - ys_j_min) / hs_i

    relative_coors = torch.cat([
        torch.log(xs_ij_delta + 0.00001),
        torch.log(ys_ij_delta + 0.00001),
        torch.log(ws_i / ws_j),
        torch.log(hs_i / hs_j),
    ])
    return relative_coors


class LayerNorm(nn.Module):
    """Construct a layernorm module.
            We employ a residual connection around each of the two sub-layers, followed by layer normalization.
        
        It is always directly utilized by the SublayerConnection.
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class CrossModalNodeEncoderLayer(nn.Module):
    "CrossModalTaskEncoderLayer contains cross-modal is made up of corss-attn and feed forward (defined below)"

    def __init__(self, size, node_feed_forward, dropout):
        super(CrossModalNodeEncoderLayer, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.node_feed_forward = node_feed_forward
        self.ff_sublayer = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, visual_nodes_feas, text_phrases_feas):
        """[Forward the specific cross-modal task encoder layer]

        Args:
            visual_nodes_feas ([torch.tensor]): [a tensor with shape <batch_size, number_of_heads, num_nodes, d_k>]
            text_phrases_feas ([torch.tensor]): [a tensor with shape <a tensor with shape <batch_size, number_of_heads, num_nodes, d_k>]

        Returns:
            encoded_node_features [torch.tensor]: [a tensor with shape <batch_size, number_of_heads, num_nodes, d_k>]
        """
        # print("CrossModalNodeEncoderLayer visual_nodes_feas: ", visual_nodes_feas.shape)
        # print("CrossModalNodeEncoderLayer visual_nodes_feas: ", text_phrases_feas.shape)

        fused_node_features = visual_nodes_feas + self.dropout(
            self.norm(text_phrases_feas))

        if self.node_feed_forward is not None:
            encoded_node_features = self.ff_sublayer(fused_node_features,
                                                     self.node_feed_forward)

        return encoded_node_features


def cross_modal_context_attention(node_query,
                                  node_key,
                                  node_value,
                                  text_query,
                                  text_key,
                                  text_value,
                                  mask=None,
                                  dropout=None):
    """[summary]

    Args:
        node_query ([torch.tensor]): [a tensor with shape <batch_size, number_of_heads, num_nodes, d_k>]
        node_key ([torch.tensor]): [a tensor with shape <batch_size, number_of_heads, num_nodes, d_k>]
        node_value ([torch.tensor]): [a tensor with shape <batch_size, number_of_heads, num_nodes, d_k>]
        text_query ([torch.tensor]): [a tensor with shape <batch_size, number_of_heads, num_phrases, d_k>]
        text_key ([torch.tensor]): [a tensor with shape <batch_size, number_of_heads, num_phrases, d_k>]
        text_value ([torch.tensor]): [a tensor with shape <batch_size, number_of_heads, num_phrases, d_k>]
        mask ([torch.tensor], optional): [a tensor with shape <batch_size, 1, 1, num_phrases>]. 
                                            Defaults to None. This is the mask for the text part!!!!!!!
        dropout ([type], optional): [description]. Defaults to None.

       Note: The query, key, value are computed by linear projections. This operation makes content in batch from
                 d_model => h x d_k, where d_model is the embedding dimension output from the position encoding, h
                 is the number of head (multihead), d_k is computed by d_model / h. 
            The position encoding will not change the size of input, because it computes as: x + posi_encode(x)
            d_k here is the common_features_dim
    Returns:
        _ 
        attn_text2visual_concepts_fs [torch.tensor]: [a tensor with shape <batch_size, number_of_heads, num_nodes, d_k>]
        v2t_scores [torch.tensor]: [a tensor with shape <batch_size, number_of_heads, num_phrases, num_nodes>]
        _
    """
    d_k = node_query.size(-1)
    assert d_k == node_key.size(-1) == node_value.size(-1) == text_query.size(
        -1)
    assert text_query.size(-1) == text_key.size(-1) == text_value.size(-1)

    # compute the visual to text attention -
    #   output scores with shape <batch_size, number_of_heads, num_text_phrases, num_visual_nodes>
    v2t_scores = torch.matmul(text_query, node_key.transpose(-2, -1)) \
                                / math.sqrt(d_k)
    # compute the text to visual attention -
    #   output with shape <batch_size, number_of_heads, num_visual_nodes, num_text_phrases>
    t2v_scores = torch.matmul(node_query, text_key.transpose(-2, -1)) \
                                / math.sqrt(d_k)

    if mask is not None:  # the text mask is only added to the text part!!!!!!
        t2v_scores = t2v_scores.masked_fill(mask == 0, -1e9)

    # v2t_p_attn = F.softmax(v2t_scores, dim=-1)
    t2v_p_attn = F.softmax(t2v_scores, dim=-1)

    if mask is not None:
        v2t_mask = mask.transpose(
            -2, -1
        )  # convert the mask to <batch_size, number_of_heads, num_item, 1>
        v2t_p_attn = v2t_p_attn.masked_fill(v2t_mask == 0,
                                            0)  # remove the text padding

    if dropout is not None:
        # v2t_p_attn = dropout(v2t_p_attn)
        t2v_p_attn = dropout(t2v_p_attn)

    # attn_visual2text_fs = torch.matmul(
    #     v2t_p_attn, node_value)  # merge the visual concept to text part
    attn_text2visual_fs = torch.matmul(
        t2v_p_attn, text_value)  # merge the text concepts for visual part

    # print("cross_modal_concept_attention t2v_p_attn: ", t2v_p_attn.shape)
    # print("cross_modal_concept_attention v2t_p_attn: ", v2t_p_attn.shape)
    # print("cross_modal_concept_attention attn_visual_concepts_fs: ", attn_visual2text_fs.shape)
    # print("cross_modal_concept_attention attn_text_concepts_fs: ", attn_text2visual_concepts_fs.shape)

    return _, attn_text2visual_fs, v2t_scores, _


class CrossModalContextMultiHeadedAttention(nn.Module):
    def __init__(self, com_dim, dropout=0.1):
        """[Take in model size and number of heads]

        Args:
            num_head ([int]): [number of heads of the attention]
            com_dim ([int]): [the required dimension of query, key, valye of the visual part]

            dropout (float, optional): [description]. Defaults to 0.1.

            Note: com_dim % h should be zero, thus, query, value and key size in each attention head are only com_dim / h
                    Thus, each attention head concentrates on part of the computation

                Here we map the multimodal data to the same dimension com_dim, and utilize the multihead attn
        """
        super(CrossModalContextMultiHeadedAttention, self).__init__()

        self.common_features_dim = com_dim
        # for each query/key/value, the layer is constructed as
        #   the common_space_layer that maps the input to d_model
        self.v_layers = clones(nn.Linear(com_dim, com_dim), 1)
        self.t_layers = clones(nn.Linear(com_dim, com_dim), 1)

        self.t2v_attn = None
        self.v2t_attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                visual_query,
                visual_key,
                visual_value,
                text_query,
                text_key,
                text_value,
                mask=None):  # query, key and value are the same of x
        """[summary]

        Args:
            visual_query ([torch.tensor]): [a tensor with shape <batch_size, num_of_nodes, num_of_heads, num_of_visual_features>]
            visual_key ([torch.tensor]): [a tensor with shape <batch_size, num_of_nodes, num_of_heads, num_of_visual_features>]
            visual_value ([torch.tensor]): [a tensor with shape <batch_size, num_of_nodes, num_of_heads, num_of_visual_features>]
            text_query ([torch.tensor]): [a tensor with shape <batch_size, num_of_nodes, num_of_heads, num_of_text_features>]
            text_key ([torch.tensor]): [a tensor with shape <batch_size, num_of_nodes, num_of_heads, num_of_text_features>]
            text_value ([torch.tensor]): [a tensor with shape <batch_size, num_of_nodes, num_of_heads, num_of_text_features>]
            mask ([torch.tensor], optional): [a tensor with shape <batch_size, num_of_nodes, 1, num_item_text>]. Defaults to None.

        Returns:
            attn_visual_features [torch.tensor]: [description]
            attn_text_features [torch.tensor]: [description]
        """

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(
                1
            )  # Convert the shape of the mask to <batch_size * task_samples, 1, 1, num_item_text>

        num_batches = visual_query.size(0)
        num_heads = visual_query.size(2)
        # 1-visual) Do all the linear projections in batch from d_model => num_heads x d_k  = com_dim
        visual_query, visual_key, visual_value = \
            [l(x).view(num_batches, -1, num_heads, self.common_features_dim).transpose(1, 2)
             for l, x in zip(self.v_layers, (visual_query, visual_key, visual_value))]

        text_query, text_key, text_value = \
            [l(x).view(num_batches, -1, num_heads, self.common_features_dim).transpose(1, 2)
             for l, x in zip(self.t_layers, (text_query, text_key, text_value))]

        # 2) Apply attention on all the projected vectors in batch.
        _, attn_text2visual_fs, \
            self.v2t_scores, _= cross_modal_concept_attention(visual_query, visual_key, visual_value,
                                            text_query, text_key, text_value,
                                            mask=mask,
                                            dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # attn_visual_concepts_fs = attn_visual_concepts_fs.transpose(1, 2).contiguous() \
        #                             .view(num_batches, -1, num_heads * self.common_features_dim)
        attn_text2visual_fs = attn_text2visual_fs.transpose(1, 2).contiguous() \
                                    .view(num_batches, -1, num_heads * self.common_features_dim)

        # Here is one confusion, we do not know whether need additional layer to further map the features

        # attn_visual_features, attn_text_features = [l(x) for l, x in zip(self.additional_layers,
        #                                                                 (attn_visual_features, attn_text_features))]

        # print("CrossModalConceptMultiHeadedAttention self.t2v_attn: ", self.t2v_attn.shape)
        # print("CrossModalConceptMultiHeadedAttention self.v2t_attn: ", self.v2t_attn.shape)
        # print("CrossModalConceptMultiHeadedAttention attn_visual_concepts_fs: ", attn_visual_concepts_fs.shape)
        # print("CrossModalConceptMultiHeadedAttention attn_text_concepts_fs: ", attn_text_concepts_fs.shape)

        return attn_text2visual_fs, self.v2t_scores


class LayerNorm(nn.Module):
    """Construct a layernorm module.
            We employ a residual connection around each of the two sub-layers, followed by layer normalization.
        
        It is always directly utilized by the SublayerConnection.
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class TwoDTransformationHead(nn.Module):
    def __init__(self, input_dims=1024, coefficients_dim=4):
        super(TwoDTransformationHead, self).__init__()
        self.coefficients_dim = 4

        self.linear_proj = nn.Linear(input_dims, 256, bias=True)
        self.relu = nn.LeakyReLU(0.2)
        self.opt_linear_proj = nn.Linear(256, 4, bias=False)

    def obtain_robust_coefficients(self, raw_coefficients):
        s1 = torch.index_select(raw_coefficients, 1, 0)
        s2 = torch.index_select(raw_coefficients, 1, 1)
        r1 = torch.index_select(raw_coefficients, 1, 2)
        r1 = torch.index_select(raw_coefficients, 1, 3)

        robust_s1 = s1 * 0.5 + 1
        robust_s2 = s2 * 0.5 + 1

        robust_r1 = r1 * 0.2
        robust_r2 = r2 * 0.2
        robust_coefficients = torch.concat(
            (robust_s1, robust_s2, robust_r1, robust_r2), 1)
        return robust_coefficients

    def forward(self, nodes_features):
        inter_feas = self.relu(self.linear_proj(nodes_features))
        raw_predicted_opts = torch.nn.functional.tanh(
            self.opt_linear_proj(inter_feas))

        return self.obtain_robust_coefficients(raw_predicted_opts)

    def normalize_coors(coors, h, w):
        coors = coors.int()
        scales = [h, w, h, w]
        normed_coors = torch.div(coors, scales)

        return normed_coors

    def get_expected_trans_coeffs(self, boxes, groundtruth_boxes, Hs, Ws):
        ''' Getting the expected trans coeffs for each box 
            Args:
                boxes: current boxes (without applying transformation) [num_boxes, 4]
                groundtruth_boxes: the traget region [num_phrases, 4]

            output:
                boxes_phs_gt_epsilons: tensor with shape [num_phrases, num_anchors, 4]
        '''

        normed_boxes = self.normalize_coors(boxes, Hs, Ws)
        normed_groundtruth_boxes = self.normalize_coors(
            groundtruth_boxes, Hs, Ws)

        normed_groundtruth_boxes = normed_groundtruth_boxes.float()
        num_boxes = normed_groundtruth_boxes.shape[0]
        num_boxes = torch.reshape(num_boxes, ())

        # print('normed_boxes: ', normed_boxes)
        # print('normed_groundtruth_boxes: ', normed_groundtruth_boxes)
        def ph_trans_coeff(ph_gt_box):
            # [gt_ymin, gt_xmin, gt_ymax, gt_xmax]
            ph_gt_miny = ph_gt_box[0]
            ph_gt_minx = ph_gt_box[1]
            ph_gt_maxy = ph_gt_box[2]
            ph_gt_maxx = ph_gt_box[3]

            boxes_ymins = torch.index_select(normed_boxes, 1, 0)
            boxes_xmins = torch.index_select(normed_boxes, 1, 1)
            boxes_ymaxs = torch.index_select(normed_boxes, 1, 2)
            boxes_xmaxs = torch.index_select(normed_boxes, 1, 3)

            epsilon_r1 = torch.div(
                ph_gt_miny * boxes_ymaxs - boxes_ymins * ph_gt_maxy,
                boxes_ymaxs - boxes_ymins)
            epsilon_r2 = torch.div(
                ph_gt_minx * boxes_xmaxs - boxes_xmins * ph_gt_maxx,
                boxes_xmaxs - boxes_xmins)

            epsilon_s2 = torch.div(ph_gt_minx - ph_gt_maxx,
                                   boxes_xmins - boxes_xmaxs)
            epsilon_s1 = torch.div(ph_gt_miny - ph_gt_maxy,
                                   boxes_ymins - boxes_ymaxs)

            # [num_boxes, 4]
            epsilon = torch.concat(
                [epsilon_s1, epsilon_s2, epsilon_r1, epsilon_r2], dim=1)

            return epsilon

        boxes_phs_gt_epsilons = torch.zeros_like(normed_boxes)
        for i in range(num_boxes):
            box_ = normed_groundtruth_boxes[i]
            boxes_phs_gt_epsilons[i] = ph_trans_coeff(box_)
        # the phs transform coefficients for each box
        # so get [num_phs, num_boxes, 4]

        return boxes_phs_gt_epsilons

    def operate_transformation(self, bounding_boxes, trans_coefficients, Hs,
                               Ws):
        ''' transfrom the boxes acording to the predicted coefficients 

            Args:
                boxes: boxes with shape [num_boxes, 4] [y_min, x_min, y_max, x_max] 
                predicted_tras_coefficients: predicted transform coefficients [num_boxes, 4] 4: 
                                            [s1, s2, r1, r2]
        '''

        normed_boxes = self.normalize_coors(bounding_boxes, Hs, Ws)
        normed_boxes = normed_boxes.float()
        num_boxes = normed_boxes.shape[0]
        boxes_num_range = torch.range(num_boxes)

        # construct the transform matrix
        def construct_transform_matrix(transform_coefficient):
            # input is a tensor with shape [4] --> [s1, s2, r1, r2]
            transform_coefficient = torch.reshape(transform_coefficient,
                                                  (-1, ))
            base_r_mt = torch.eye(3)
            coe_r = torch.index_select(transform_coefficient, 1, [2, 3])
            coe_r = torch.reshape(coe_r, (-1, 1))
            paddings = torch.Tensor([[0, 1], [2, 0]])
            oper_coe_r = nn.functional.pad(coe_r, paddings, "CONSTANT")
            r_mt = base_r_mt + oper_coe_r

            # tf.matrix_diag(tf.constant([0, 0, 1], tf.float32))
            base_s_mt = torch.diag_embed(tf.constant([0, 0, 1]))
            coe_s = torch.index_select(transform_coefficient, 1, [0, 1])
            diag_coe_s = torch.diag_embed(coe_s)
            paddings = torch.Tensor([[0, 1], [0, 1]])
            oper_diag_coe_s = nn.functional.pad(diag_coe_s, paddings,
                                                "CONSTANT")
            s_mt = base_s_mt + oper_diag_coe_s

            transform_matrix = torch.matmul(r_mt, s_mt)

            return transform_matrix

        def box_transform(box_idx):
            box_idx = box_idx.int()
            # get coordinate of the box [ymin, xmin, ymax, xmax]
            box_coor = torch.index_select(normed_boxes, 0, box_idx)
            box_trans_coe = torch.index_select(predicted_tras_coefficients, 0,
                                               box_idx)

            top_left_pos = torch.index_select(box_coor, 1, [0, 1])
            top_left_pos = torch.reshape(top_left_pos, [-1, 1])
            top_left_pos_h = nn.functional.pad(top_left_pos, [[0, 1], [0, 0]],
                                               "CONSTANT",
                                               value=1)

            bottom_right_pos = torch.index_select(box_coor, 1, [2, 3])
            bottom_right_pos = torch.reshape(bottom_right_pos, [-1, 1])
            bottom_right_pos_h = nn.functional.pad(bottom_right_pos,
                                                   [[0, 1], [0, 0]],
                                                   "CONSTANT",
                                                   value=1)

            transform_matrix = construct_transform_matrix(box_trans_coe)

            transformed_top_left_pos = torch.matmul(transform_matrix,
                                                    top_left_pos_h)
            # transformed_top_left_pos, [0, 0], [2, 1])
            transformed_top_left_pos = torch.index_select(
                transformed_top_left_pos, 0, [0, 1])
            transformed_top_left_pos = torch.index_select(
                transformed_top_left_pos, 1, [0])
            transformed_top_left_pos = torch.reshape(transformed_top_left_pos,
                                                     [1, 2])

            transformed_bottom_right_pos = torch.matmul(
                transform_matrix, bottom_right_pos_h)
            # transformed_bottom_right_pos, [0, 0], [2, 1]
            transformed_bottom_right_pos = torch.index_select(
                transformed_bottom_right_pos, 0, [0, 1])
            transformed_bottom_right_pos = torch.index_select(
                transformed_bottom_right_pos, 1, [0])
            transformed_bottom_right_pos = torch.reshape(
                transformed_bottom_right_pos, [1, 2])

            transformed_box = torch.concat(
                [transformed_top_left_pos, transformed_bottom_right_pos],
                dim=1)
            transformed_box = torch.reshape(transformed_box, [
                -1,
            ])
            return transformed_box

        normed_transformed_boxes = torch.zeros_like(normed_boxes)
        for i in range(num_boxes):
            box_idx = boxes_num_range[i]
            normed_transformed_boxes[i] = box_transform(box_idx)

        # print(ok)
        return normed_transformed_boxes