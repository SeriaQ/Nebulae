#!/usr/bin/env python
'''
dcvae
Created by Seria at 2022/11/8 8:15 PM
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
from .architect import ConvE, DeconvD


class DCVAE(dock.Craft):
    def __init__(self, in_shape, scope='DCVAE'):
        super(DCVAE, self).__init__(scope)
        H, W, C = in_shape
        latent_dim = H//16 * W//16 * 128
        self.E = ConvE(in_shape)
        self.D = DeconvD(in_shape, latent_dim)

    def run(self, x):
        mu, lv, z = self.E(x)
        y = self.D(z)
        return mu, lv, y