#!/usr/bin/env python
'''
Created by Seria at 02/11/2018 2:50 PM
Email: zzqsummerai@yeah.net

                    _ooOoo_
                  o888888888o
                 o88`_ . _`88o
                 (|  0   0  |)
                 O \   。   / O
              _____/`-----‘\_____
            .’   \||  _ _  ||/   `.
            |  _ |||   |   ||| _  |
            |  |  \\       //  |  |
            |  |    \-----/    |  |
             \ .\ ___/- -\___ /. /
         ,--- /   ___\<|>/___   \ ---,
         | |:    \    \ /    /    :| |
         `\--\_    -. ___ .-    _/--/‘
   ===========  \__  NOBUG  __/  ===========
   
'''
# -*- coding:utf-8 -*-
from .decorator import Timer, SPST
from .utility import hotvec2mtx, mtx2hotvec, den2spa, spa2den, rand_trunc_norm, parse_cfg, record_cfg, \
                    byte2arr, rgb2y, join_imgs, plot_in_one, ver2num, GPUtil, sprawl


import torch
import random
import numpy as np
def destine(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)