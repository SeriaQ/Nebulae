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
from .architect import BigG, BigD, BN


class Discriminator(dock.Craft):
    def __init__(self, in_shape, base_chs, norm_fn, attention, spec_norm, w_init, scope='DSC'):
        super(Discriminator, self).__init__(scope)
        H, W, C = in_shape
        min_size = min(H, W)
        factor = {128: base_chs/16}
        self.backbone = BigD(in_shape, base_chs, norm_fn, attention, spec_norm, w_init)
        if spec_norm:
            self.cls = dock.SN(dock.Dense(int(H * W * factor[min_size]), 1))
        else:
            self.cls = dock.Dense(int(H * W * factor[min_size]), 1)

    def run(self, x):
        x = self.backbone(x)
        self['out'] = self.cls(x)

        return self['out']



class BigGAN(dock.Craft):
    def __init__(self, in_shape, latent_dim=128, base_chs=64, norm_fn=BN, attention=False,
                 spec_norm=False, w_init=dock.XavierNorm(), scope='BIGGAN'):
        super(BigGAN, self).__init__(scope)
        self.G = BigG(in_shape, latent_dim, base_chs, norm_fn, attention, spec_norm, w_init)
        self.D = Discriminator(in_shape, base_chs, norm_fn, attention, spec_norm, w_init)

    def run(self):
        pass