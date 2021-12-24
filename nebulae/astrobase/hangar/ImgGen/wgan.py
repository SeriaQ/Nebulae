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
from .architect import MLPG, MLPD


class WMLPD(dock.Craft):
    def __init__(self, in_shape, latent_dim, scope='WMLPD'):
        super(WMLPD, self).__init__(scope)
        self.backbone = MLPD(in_shape, latent_dim)
        self.cls = dock.Dense(latent_dim, 1)
        self.clipper = dock.Clip(intrinsic=True)

    def run(self, x):
        self['input'] = x
        self['out'] = self.cls(self.backbone(self['input']))

        return self['out']

    def clip(self, ranges):
        for p in self.params():
            _ = self.clipper(p, ranges)


class WGAN(dock.Craft):
    def __init__(self, in_shape, latent_dim=128, scope='WGAN'):
        super(WGAN, self).__init__(scope)
        self.G = MLPG(in_shape, latent_dim)
        self.D = WMLPD(in_shape, latent_dim)

    def run(self):
        pass