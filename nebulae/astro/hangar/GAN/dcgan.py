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
from .architect import ConvG, ConvD

from math import ceil

class Discriminator(dock.Craft):
    def __init__(self, in_shape, scope='DSC'):
        super(Discriminator, self).__init__(scope)
        H, W, C = in_shape
        self.backbone = ConvD(in_shape)
        self.cls = dock.Dense(ceil(H / 16) * ceil(W / 16) * 128, 1)

    def run(self, x):
        x = self.backbone(x)
        self['out'] = self.cls(x)

        return self['out']



class DCGAN(dock.Craft):
    def __init__(self, in_shape, latent_dim=128, scope='DCGAN'):
        super(DCGAN, self).__init__(scope)
        self.G = ConvG(in_shape, latent_dim)
        self.D = Discriminator(in_shape)

    def run(self):
        pass