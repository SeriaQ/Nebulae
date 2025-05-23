#!/usr/bin/env python
# -*- coding:utf-8 -*-
from ... import dock
from .architect import DeconvG, ConvD

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

class FCGAN(dock.Craft):
    def __init__(self, in_shape, latent_dim=128, scope='FCGAN'):
        super(FCGAN, self).__init__(scope)
        self.G = DeconvG(in_shape, latent_dim)
        self.D = Discriminator(in_shape)

    def run(self):
        pass