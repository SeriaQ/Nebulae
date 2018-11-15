#!/usr/bin/env python
'''
utility
Created by Seria at 14/11/2018 8:33 PM
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

import numpy as np

def toOneHot(labels, nclass, on_value=1, off_value=0):
    batch_size = len(labels)
    # initialize one-hot labels
    one_hot = off_value * np.ones((batch_size * nclass))
    indices = []
    if isinstance(labels[0], str):
        for b in range(batch_size):
            indices += [int(s) + b * nclass for s in labels[b].split(' ')]
    else: # labels is a nested list
        for b in range(batch_size):
            indices += [l + b * nclass for l in labels[b]]
    one_hot[indices] = on_value
    return np.reshape(one_hot, (batch_size, nclass))