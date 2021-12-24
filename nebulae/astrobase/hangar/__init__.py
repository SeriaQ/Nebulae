#!/usr/bin/env python
'''
__init__
Created by Seria at 23/11/2018 10:30 AM
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
from .Classical.vgg import VGG_16
from .Classical.resnet import Resnet_V2_50, Resnet_V2_101, Resnet_V2_152
# from .senet import SE_RESNET_V2_50, SE_RESNET_V2_101, SE_RESNET_V2_152
from .ImgGen.gan import GAN
from .ImgGen.dcgan import DCGAN
from .ImgGen.fcgan import FCGAN
from .ImgGen.resgan import ResGAN
from .ImgGen.biggan import BigGAN
from .ImgGen.wgan import WGAN
from .ImgGen.wgan_div import WGANDiv
from .ImgGen.infogan import InfoGAN
from .Seq2Seq.architect import RNNE, BiRNNE, RNND, AttnRNND, LSTME, LSTMD, AttnLSTMD

BN = 20
CBN = 21
IN = 22
CIN = 23
LN = 24