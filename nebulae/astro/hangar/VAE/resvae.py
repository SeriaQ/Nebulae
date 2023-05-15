#!/usr/bin/env python
'''
vae
Created by Seria at 2022/11/3 4:06 PM
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