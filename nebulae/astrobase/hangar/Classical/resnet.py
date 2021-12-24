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



class Bottleneck(dock.Craft):
    def __init__(self, in_shape, neck_chs, body_chs, stride=1, dsample=False, scope='BOTTLENECK'):
        super(Bottleneck, self).__init__(scope)
        H, W, C = in_shape
        self.body_chs = body_chs
        self.conv_1 = dock.Conv(C, neck_chs, (1, 1))
        self.bn_1 = dock.BN(neck_chs, dim=2)
        pad = dock.autoPad((H, W), (3, 3), stride=stride)
        self.conv_2 = dock.Conv(neck_chs, neck_chs, (3, 3), stride=stride, padding=pad)
        self.bn_2 = dock.BN(neck_chs, dim=2)
        self.conv_3 = dock.Conv(neck_chs, self.body_chs, (1, 1))
        self.bn_3 = dock.BN(self.body_chs, dim=2)
        self.relu = dock.Relu()

        self.dsample = dsample
        if dsample:
            self.ds_conv = dock.Conv(C, self.body_chs, (1, 1), stride)
            self.ds_bn = dock.BN(self.body_chs, dim=2)

    def run(self, x):
        self['identity'] = x

        y = self.conv_1(x)
        y = self.bn_1(y)
        y = self.relu(y)

        y = self.conv_2(y)
        y = self.bn_2(y)
        y = self.relu(y)

        y = self.conv_3(y)
        y = self.bn_3(y)

        if self.dsample:
            self['identity'] = self.ds_bn(self.ds_conv(x))

        y += self['identity']
        y = self.relu(y)

        return y


class ResBlock(dock.Craft):
    def __init__(self, in_shape, nchs, nblock, stride=1, width_multp=4, scope='RESBLOCK'):
        super(ResBlock, self).__init__(scope)
        H, W, C = in_shape
        self.nblock = nblock
        self.layer_0 = Bottleneck(in_shape, nchs, nchs*width_multp, stride, dsample=True)
        for i in range(1, nblock):
            setattr(self, 'layer_%d'%i, Bottleneck((H, W, nchs * width_multp), nchs, nchs * width_multp))

    def run(self, x):
        self['input'] = x
        for i in range(self.nblock):
            l = getattr(self, 'layer_%d'%i)
            x = l(x)

        return x


class Resnet_V2(dock.Craft):
    def __init__(self, in_shape, nblocks, scope='RESNET_V2'):
        super(Resnet_V2, self).__init__(scope)
        H, W, C = in_shape
        self.H, self.W, self.C = H, W, C

        pad = dock.autoPad((H, W), (7, 7), 2)
        self.conv_0 = dock.Conv(C, 64, (7, 7), stride=2, padding=pad)
        self.bn_0 = dock.BN(64, dim=2)
        self.relu = dock.Relu()
        pad = dock.autoPad((ceil(H/2), ceil(W/2)), (3, 3), 2)
        self.mpool = dock.MaxPool((3, 3), padding=pad)

        self.nblocks = nblocks
        ds_power = 2
        width_multp = 4
        ichs = 64
        nchs = 64
        for i, nblk in enumerate(nblocks):
            if i == 0:
                stride = 1
            else:
                stride = 2
                ds_power += 1
            setattr(self, 'layer_%d'%i, ResBlock((ceil(H/2**ds_power), ceil(W/2**ds_power), ichs),
                                                 nchs, nblk, stride, width_multp))
            ichs *= 4 if i==0 else 2
            nchs *= 2

        self.gap = dock.AvgPool((-1, -1))

    def run(self, x):
        self['input'] = x
        y = self.conv_0(self['input'])
        y = self.bn_0(y)
        y = self.relu(y)
        y = self.mpool(y)

        for i in range(len(self.nblocks)):
            blk = getattr(self, 'layer_%d'%i)
            y = blk(y)

        self['out'] = self.gap(y)
        return self['out']



class Resnet_V2_50(dock.Craft):
    def __init__(self, in_shape, scope='RESNET_V2_50'):
        super(Resnet_V2_50, self).__init__(scope)
        self.backbone = Resnet_V2(in_shape, [3, 4, 6, 3])

    def run(self, x):
        self['input'] = x
        self['out'] = self.backbone(self['input'])
        return self['out']


class Resnet_V2_101(dock.Craft):
    def __init__(self, in_shape, scope='RESNET_V2_101'):
        super(Resnet_V2_101, self).__init__(scope)
        self.backbone = Resnet_V2(in_shape, [3, 4, 23, 3])

    def run(self, x):
        self['input'] = x
        self['out'] = self.backbone(self['input'])
        return self['out']


class Resnet_V2_152(dock.Craft):
    def __init__(self, in_shape, scope='RESNET_V2_152'):
        super(Resnet_V2_152, self).__init__(scope)
        self.backbone = Resnet_V2(in_shape, [3, 8, 36, 3])

    def run(self, x):
        self['input'] = x
        self['out'] = self.backbone(self['input'])
        return self['out']