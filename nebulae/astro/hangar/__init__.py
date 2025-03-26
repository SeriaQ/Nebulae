#!/usr/bin/env python
# -*- coding:utf-8 -*-
from .Classic.vgg import VGG_16
from .Classic.resnet import Resnet_V2_50, Resnet_V2_101, Resnet_V2_152

from .GAN.gan import GAN
from .GAN.dcgan import DCGAN
# from .GAN.fcgan import FCGAN
from .GAN.resgan import ResGAN
# from .GAN.biggan import BigGAN
# from .GAN.wgan import WGAN
# from .GAN.wgan_div import WGANDiv
# from .GAN.infogan import InfoGAN

from .VAE.vae import VAE
from .VAE.dcvae import DCVAE
from .VAE.resvae import ResVAE
from .VAE.vqvae import VQVAE

# from .Seq2Seq.architect import RNNE, BiRNNE, RNND, AttnRNND, LSTME, LSTMD, AttnLSTMD

BN = 20
CBN = 21
IN = 22
CIN = 23
LN = 24