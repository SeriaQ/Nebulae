#!/usr/bin/env python
# -*- coding:utf-8 -*-
from ... import dock
from .architect import MLPE, MLPD



class VAE(dock.Craft):
    def __init__(self, in_shape, hidden_dim, latent_dim, scope='VAE'):
        super(VAE, self).__init__(scope)
        self.E = MLPE(in_shape, hidden_dim, latent_dim)
        self.D = MLPD(in_shape, hidden_dim, latent_dim)

    def run(self, x):
        mu, lv, z = self.E(x)
        y = self.D(z)
        return mu, lv, y