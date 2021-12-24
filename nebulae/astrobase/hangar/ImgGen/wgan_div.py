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
    def __init__(self, in_shape, latent_dim, k, p, scope='WMLPD'):
        super(WMLPD, self).__init__(scope)
        self.k = k
        self.p = p
        self.backbone = MLPD(in_shape, latent_dim)
        self.cls = dock.Dense(latent_dim, 1)

        self.sum = dock.Sum()
        self.grad = dock.Grad()
        self.flat = dock.Reshape()

    def run(self, x, grad_penalty):
        self['input'] = x
        if grad_penalty:
            self['input'] = self.engine.coat(self['input'], as_const=False)

        self['out'] = self.cls(self.backbone(self['input']))

        if grad_penalty:
            # self.chain(self['input'])
            g = self.grad(self['input'], self['out'])[0]
            g = self.flat(g, (g.shape[0], -1))
            g = self.sum(g**2, -1)
            self['grad_norm'] = self.k * g ** (self.p/2)

            return self['out'], self['grad_norm']
        else:
            return self['out']



class WGANDiv(dock.Craft):
    def __init__(self, in_shape, latent_dim=128, k=2, p=6, scope='WGANDIV'):
        super(WGANDiv, self).__init__(scope)
        self.G = MLPG(in_shape, latent_dim)
        self.D = WMLPD(in_shape, latent_dim, k, p)

    def run(self):
        pass