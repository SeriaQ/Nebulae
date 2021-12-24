#!/usr/bin/env python
'''
garage
Created by Seria at 03/01/2019 8:32 PM
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
from ... import dock

from math import ceil

class VGG_16(dock.Craft):
    def __init__(self, in_shape, p_drop=0.5, scope='VGG_16'):
        super(VGG_16, self).__init__(scope)
        self.p_drop = p_drop
        H, W, C = in_shape
        self.relu = dock.Relu()
        pad = dock.autoPad((H, W), (3, 3))
        self.conv_1_1 = dock.Conv(C, 8, (3, 3), padding=pad)
        self.conv_1_2 = dock.Conv(8, 64, (3, 3), padding=pad)
        pad = dock.autoPad((H, W), (2, 2), stride=2)
        self.mpool_1 = dock.MaxPool((2, 2), padding=pad)

        pad = dock.autoPad((ceil(H / 2), ceil(W / 2)), (3, 3))
        self.conv_2_1 = dock.Conv(64, 128, (3, 3), padding=pad)
        self.conv_2_2 = dock.Conv(128, 128, (3, 3), padding=pad)
        pad = dock.autoPad((ceil(H / 2), ceil(W / 2)), (2, 2), stride=2)
        self.mpool_2 = dock.MaxPool((2, 2), padding=pad)

        pad = dock.autoPad((ceil(H / 4), ceil(W / 4)), (3, 3))
        self.conv_3_1 = dock.Conv(128, 256, (3, 3), padding=pad)
        self.conv_3_2 = dock.Conv(256, 256, (3, 3), padding=pad)
        self.conv_3_3 = dock.Conv(256, 256, (3, 3), padding=pad)
        pad = dock.autoPad((ceil(H / 4), ceil(W / 4)), (2, 2), stride=2)
        self.mpool_3 = dock.MaxPool((2, 2), padding=pad)

        pad = dock.autoPad((ceil(H / 8), ceil(W / 8)), (3, 3))
        self.conv_4_1 = dock.Conv(256, 512, (3, 3), padding=pad)
        self.conv_4_2 = dock.Conv(512, 512, (3, 3), padding=pad)
        self.conv_4_3 = dock.Conv(512, 512, (3, 3), padding=pad)
        pad = dock.autoPad((ceil(H / 8), ceil(W / 8)), (2, 2), stride=2)
        self.mpool_4 = dock.MaxPool((2, 2), padding=pad)

        pad = dock.autoPad((ceil(H / 16), ceil(W / 16)), (3, 3))
        self.conv_5_1 = dock.Conv(512, 512, (3, 3), padding=pad)
        self.conv_5_2 = dock.Conv(512, 512, (3, 3), padding=pad)
        self.conv_5_3 = dock.Conv(512, 512, (3, 3), padding=pad)
        pad = dock.autoPad((ceil(H / 16), ceil(W / 16)), (2, 2), stride=2)
        self.mpool_5 = dock.MaxPool((2, 2), padding=pad)

        self.dropout = dock.Dropout(0.2, dim=2)
        self.glob_conv_1 = dock.Conv(512, 4096, (ceil(H / 32), ceil(W / 32)))
        self.glob_conv_2 = dock.Conv(4096, 4096, (1, 1))



    def run(self, x):
        self['input'] = x
        c1 = self.conv_1_1(self['input'])
        c1 = self.relu(c1)
        c1 = self.conv_1_2(c1)
        c1 = self.relu(c1)
        self['conv_1'] = self.mpool_1(c1)

        c2 = self.conv_2_1(self['conv_1'])
        c2 = self.relu(c2)
        # c2 = self.conv_2_2(c2)
        # c2 = self.relu(c2)
        self['conv_2'] = self.mpool_2(c2)

        c3 = self.conv_3_1(self['conv_2'])
        c3 = self.relu(c3)
        # c3 = self.conv_3_2(c3)
        # c3 = self.relu(c3)
        # c3 = self.conv_3_3(c3)
        # c3 = self.relu(c3)
        self['conv_3'] = self.mpool_3(c3)

        c4 = self.conv_4_1(self['conv_3'])
        c4 = self.relu(c4)
        # c4 = self.conv_4_2(c4)
        # c4 = self.relu(c4)
        # c4 = self.conv_4_3(c4)
        # c4 = self.relu(c4)
        self['conv_4'] = self.mpool_4(c4)

        c5 = self.conv_5_1(self['conv_4'])
        c5 = self.relu(c5)
        # c5 = self.conv_5_2(c5)
        # c5 = self.relu(c5)
        # c5 = self.conv_5_3(c5)
        # c5 = self.relu(c5)
        self['conv_5'] = self.mpool_5(c5)

        gc = self.glob_conv_1(self['conv_5'])
        gc = self.dropout(gc)
        gc = self.glob_conv_2(gc)
        self['gconv'] = self.dropout(gc)

        return self['gconv']