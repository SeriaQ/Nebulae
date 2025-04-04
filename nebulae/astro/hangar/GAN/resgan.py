#!/usr/bin/env python
# -*- coding:utf-8 -*-
from ... import dock
from .architect import ResG, ResD, BN


class Discriminator(dock.Craft):
    def __init__(self, in_shape, base_chs, norm_fn, attention, spec_norm, w_init, scope='DSC'):
        super(Discriminator, self).__init__(scope)
        H, W, C = in_shape
        min_size = min(H, W)
        factor = {64: base_chs // 32,
                  128: base_chs // 16,
                  256: base_chs // 64}
        self.backbone = ResD(in_shape, base_chs, norm_fn, attention, spec_norm, w_init)
        if spec_norm:
            self.cls = dock.SN(dock.Dense(int(H * W * factor[min_size]), 1))
        else:
            self.cls = dock.Dense(int(H * W * factor[min_size]), 1)

    def run(self, x):
        x = self.backbone(x)
        self['out'] = self.cls(x)

        return self['out']



class ResGAN(dock.Craft):
    def __init__(self, in_shape, latent_dim=128, base_chs=64, norm_fn=BN, attention=False,
                 spec_norm=False, w_init=dock.XavierNorm(), scope='RESGAN'):
        super(ResGAN, self).__init__(scope)
        self.G = ResG(in_shape, latent_dim, base_chs, norm_fn, attention, spec_norm, w_init)
        self.D = Discriminator(in_shape, base_chs, norm_fn, attention, spec_norm, w_init)

    def run(self):
        pass