#!/usr/bin/env python
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