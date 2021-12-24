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
from math import ceil
from ... import dock
from .architect import ConvG, ConvD, ResG, ResD, BN



class Discriminator(dock.Craft):
    # def __init__(self, in_shape, category_dim, code_dim, scope='DSC'):
    def __init__(self, in_shape, category_dim, code_dim, base_chs, norm_fn, attention, spec_norm, w_init, scope='DSC'):
        super(Discriminator, self).__init__(scope)
        H, W, C = in_shape
        self.H = H
        self.W = W
        self.C = C
        self.category_dim = category_dim
        self.code_dim = code_dim
        self.sftm = dock.Sftm()

        # self.backbone = ConvD(in_shape)
        # feat_dim = ceil(H/16) * ceil(W/16) * 128
        # self.cls = dock.Dense(feat_dim, 1)
        # if category_dim > 0:
        #     self.category = dock.Dense(feat_dim, category_dim)
        # if code_dim > 0:
        #     self.code = dock.Dense(feat_dim, code_dim)

        ############# FENCE ##############
        min_size = min(H, W)
        factor = {128: base_chs / 16}
        self.backbone = ResD(in_shape, base_chs, norm_fn, attention, spec_norm, w_init)
        feat_dim = int(H * W * factor[min_size])
        if spec_norm:
            self.cls = dock.SN(dock.Dense(feat_dim, 1))
            if category_dim > 0:
                self.category = dock.SN(dock.Dense(feat_dim, category_dim))
            if code_dim > 0:
                self.code = dock.SN(dock.Dense(feat_dim, code_dim))
        else:
            self.cls = dock.Dense(feat_dim, 1)
            if category_dim > 0:
                self.category = dock.Dense(feat_dim, category_dim)
            if code_dim > 0:
                self.code = dock.Dense(feat_dim, code_dim)

    def run(self, x):
        self['input'] = x
        x = self.backbone(x)

        self['cls'] = self.cls(x)
        y = [self['cls']]
        if self.category_dim > 0:
            self['category'] = self.sftm(self.category(x))
            y.append(self['category'])
        if self.code_dim > 0:
            self['code'] = self.code(x)
            y.append(self['code'])

        if len(y) > 1:
            return tuple(y)
        else:
            return y[0]



class InfoGAN(dock.Craft):
    # def __init__(self, in_shape, category_dim=0, code_dim=0, latent_dim=128, scope='INFOGAN'):
    def __init__(self, in_shape, category_dim=0, code_dim=0, latent_dim=128, base_chs=32, norm_fn=BN,
                 attention=False, spec_norm=False, w_init=dock.XavierNorm(), scope='INFOGAN'):
        super(InfoGAN, self).__init__(scope)
        # self.G = ConvG(in_shape, category_dim + code_dim + latent_dim)
        # self.D = Discriminator(in_shape, category_dim, code_dim)
        self.G = ResG(in_shape, category_dim + code_dim + latent_dim, base_chs, norm_fn, attention, spec_norm, w_init)
        self.D = Discriminator(in_shape, category_dim, code_dim, base_chs, norm_fn, attention, spec_norm, w_init)

    def run(self):
        pass