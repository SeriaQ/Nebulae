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
from .decorator import Timer
from .utility import hotvec2mtx, mtx2hotvec, den2spa, spa2den, randTruncNorm, parseConfig, recordConfig, byte2arr, \
                    joinImg, plotInOne, ver2num, GPUtil

import os
core = os.environ.get('NEB_CORE', 'PYTORCH')
if core.upper() == 'TENSORFLOW':
    import tensorflow as tf
    import random
    import numpy as np
    def destine(seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
elif core.upper() == 'PYTORCH':
    import torch
    import random
    import numpy as np
    def destine(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
else:
    raise ValueError('NEBULAE ERROR ⨷ %s is an unsupported core.' % core)
