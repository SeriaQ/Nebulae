#!/usr/bin/env python
# -*- coding:utf-8 -*-
from ... import dock


class VGG_16(dock.Craft):
    def __init__(self, in_chs, p_drop=0.5, scope='VGG_16'):
        super(VGG_16, self).__init__(scope)
        self.p_drop = p_drop
        self.relu = dock.Relu()
        pad = dock.autopad((3, 3))
        self.conv_1_1 = dock.Conv(in_chs, 8, (3, 3), padding=pad)
        self.conv_1_2 = dock.Conv(8, 64, (3, 3), padding=pad)
        pad = dock.autopad((2, 2), stride=2)
        self.mpool_1 = dock.MaxPool((2, 2), padding=pad)

        pad = dock.autopad((3, 3))
        self.conv_2_1 = dock.Conv(64, 128, (3, 3), padding=pad)
        self.conv_2_2 = dock.Conv(128, 128, (3, 3), padding=pad)
        pad = dock.autopad((2, 2), stride=2)
        self.mpool_2 = dock.MaxPool((2, 2), padding=pad)

        pad = dock.autopad((3, 3))
        self.conv_3_1 = dock.Conv(128, 256, (3, 3), padding=pad)
        self.conv_3_2 = dock.Conv(256, 256, (3, 3), padding=pad)
        self.conv_3_3 = dock.Conv(256, 256, (3, 3), padding=pad)
        pad = dock.autopad((2, 2), stride=2)
        self.mpool_3 = dock.MaxPool((2, 2), padding=pad)

        pad = dock.autopad((3, 3))
        self.conv_4_1 = dock.Conv(256, 512, (3, 3), padding=pad)
        self.conv_4_2 = dock.Conv(512, 512, (3, 3), padding=pad)
        self.conv_4_3 = dock.Conv(512, 512, (3, 3), padding=pad)
        pad = dock.autopad((2, 2), stride=2)
        self.mpool_4 = dock.MaxPool((2, 2), padding=pad)

        pad = dock.autopad((3, 3))
        self.conv_5_1 = dock.Conv(512, 512, (3, 3), padding=pad)
        self.conv_5_2 = dock.Conv(512, 512, (3, 3), padding=pad)
        self.conv_5_3 = dock.Conv(512, 512, (3, 3), padding=pad)
        pad = dock.autopad((2, 2), stride=2)
        self.mpool_5 = dock.MaxPool((2, 2), padding=pad)

        self.dropout = dock.Dropout(0.2, dim=2)
        self.glob_conv_1 = dock.Conv(512, 4096, (1, 1))
        self.glob_apool = dock.AvgPool((-1, -1))
        self.glob_conv_2 = dock.Conv(4096, 4096, (1, 1))



    def run(self, x):
        c1 = self.conv_1_1(x)
        c1 = self.relu(c1)
        c1 = self.conv_1_2(c1)
        c1 = self.relu(c1)
        c1 = self.mpool_1(c1)

        c2 = self.conv_2_1(c1)
        c2 = self.relu(c2)
        c2 = self.conv_2_2(c2)
        c2 = self.relu(c2)
        c2 = self.mpool_2(c2)

        c3 = self.conv_3_1(c2)
        c3 = self.relu(c3)
        c3 = self.conv_3_2(c3)
        c3 = self.relu(c3)
        c3 = self.conv_3_3(c3)
        c3 = self.relu(c3)
        c3 = self.mpool_3(c3)

        c4 = self.conv_4_1(c3)
        c4 = self.relu(c4)
        c4 = self.conv_4_2(c4)
        c4 = self.relu(c4)
        c4 = self.conv_4_3(c4)
        c4 = self.relu(c4)
        c4 = self.mpool_4(c4)

        c5 = self.conv_5_1(c4)
        c5 = self.relu(c5)
        c5 = self.conv_5_2(c5)
        c5 = self.relu(c5)
        c5 = self.conv_5_3(c5)
        c5 = self.relu(c5)
        c5 = self.mpool_5(c5)

        gc = self.glob_conv_1(c5)
        gc = self.glob_apool(gc)
        gc = self.dropout(gc)
        gc = self.glob_conv_2(gc)
        y = self.dropout(gc)

        return y