#!/usr/bin/env python
# -*- coding:utf-8 -*-
from ... import dock
from .architect import ResE, ResD



class ResVAE(dock.Craft):
    def __init__(self, in_shape, hidden_dim, latent_dim, scope='RESVAE'):
        super(ResVAE, self).__init__(scope)
        self.E = ResE(in_shape, hidden_dim, latent_dim)
        self.D = ResD(in_shape, hidden_dim, latent_dim)

    def run(self, x):
        mu, lv, z = self.E(x)
        y = self.D(z)
        return mu, lv, y