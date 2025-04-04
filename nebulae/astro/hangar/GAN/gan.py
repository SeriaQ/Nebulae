#!/usr/bin/env python
# -*- coding:utf-8 -*-
from ... import dock
from .architect import MLPG, MLPD


class Discriminator(dock.Craft):
    def __init__(self, in_shape, latent_dim, scope='DSC'):
        super(Discriminator, self).__init__(scope)
        self.backbone = MLPD(in_shape, latent_dim)
        self.cls = dock.Dense(latent_dim, 1)

    def run(self, x):
        x = self.backbone(x)
        self['out'] = self.cls(x)

        return self['out']

class GAN(dock.Craft):
    def __init__(self, in_shape, latent_dim=128, scope='GAN'):
        super(GAN, self).__init__(scope)
        self.G = MLPG(in_shape, latent_dim)
        self.D = Discriminator(in_shape, latent_dim)

    def run(self):
        pass