#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
import torch.nn as nn
import math, copy, time
import enum
from torch.utils.tensorboard import SummaryWriter


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
