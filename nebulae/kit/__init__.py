#!/usr/bin/env python
# -*- coding:utf-8 -*-
from .decorator import Timer, SPST
from .utility import hotvec2mtx, mtx2hotvec, den2spa, spa2den, rand_trunc_norm, parse_cfg, record_cfg, \
                    byte2arr, rgb2y, ver2num, GPUtil, sprawl


import torch
import random
import numpy as np
def destine(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)