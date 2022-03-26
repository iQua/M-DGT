#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from models import boxes_utils


def compute_iter_aware_loss(
    iter_i,
    vc_feas,
    gh_vc_feas,
    tc_feas,
    pred_coeffs,
    gh_coeffs,
    pred_boxes,
    gh_boxes,
    ph_attns,
):
    alpha = 1 / (1 + torch.exp(-iter_i))

    v_dist = torch.norm(torch.cdist(vc_feas, gh_vc_feas, p=2))
    cm_dist = torch.norm(torch.cdist(vc_feas, tc_feas, p=2))

    ious = boxes_utils.matrix_iou(pred_coeffs, gh_boxes)

    loss1 = max([ious, v_dist, 0])
    loss2 = max([ious, cm_dist, 0])
    l_ss = loss1 + loss2

    sl_loss = torch.nn.SmoothL1Loss(pred_coeffs, gh_coeffs)

    sum_loss = sl_loss + 0.4 * l_ss

    total_loss = torch.mean(ph_attns * sum_loss)
    return total_loss