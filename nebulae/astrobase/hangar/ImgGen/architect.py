#!/usr/bin/env python
'''
architect
Created by Seria at 2020/11/18 1:42 PM
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

from math import ceil
import numpy as np


BN = 20
CBN = 21
IN = 22
CIN = 23
LN = 24



class MLPG(dock.Craft):
    def __init__(self, in_shape, latent_dim, scope='MLPG'):
        super(MLPG, self).__init__(scope)
        H, W, C = in_shape
        self.H = H
        self.W = W
        self.C = C

        self.lrelu = dock.LRelu()
        self.fc_1 = dock.Dense(latent_dim, latent_dim)
        self.fc_2 = dock.Dense(latent_dim, latent_dim * 2)
        self.bn_2 = dock.BN(latent_dim * 2, dim=1, mmnt=0.8)
        self.fc_3 = dock.Dense(latent_dim * 2, latent_dim * 4)
        self.bn_3 = dock.BN(latent_dim * 4, dim=1, mmnt=0.8)
        self.fc_4 = dock.Dense(latent_dim * 4, latent_dim * 8)
        self.bn_4 = dock.BN(latent_dim * 8, dim=1, mmnt=0.8)

        self.fc_pixel = dock.Dense(latent_dim * 8, H * W * C)
        self.tanh = dock.Tanh()
        self.rect = dock.Reshape()

    def run(self, z):
        self['latent_code'] = z
        z = self.fc_1(self['latent_code'])
        z = self.lrelu(z)

        z = self.fc_2(z)
        z = self.bn_2(z)
        z = self.lrelu(z)

        z = self.fc_3(z)
        z = self.bn_3(z)
        z = self.lrelu(z)

        z = self.fc_4(z)
        z = self.bn_4(z)
        z = self.lrelu(z)

        z = self.fc_pixel(z)
        z = self.tanh(z)
        self['fake'] = self.rect(z, (-1, self.C, self.H, self.W))

        return self['fake']


class MLPD(dock.Craft):
    def __init__(self, in_shape, latent_dim, scope='MLPD'):
        super(MLPD, self).__init__(scope)
        H, W, C = in_shape
        self.H = H
        self.W = W
        self.C = C

        self.lrelu = dock.LRelu()
        self.flat = dock.Reshape()
        self.fc_1 = dock.Dense(H * W * C, latent_dim * 2)
        self.fc_2 = dock.Dense(latent_dim * 2, latent_dim)

    def run(self, x):
        self['input'] = x
        x = self.flat(self['input'], (-1, self.C * self.H * self.W))
        x = self.fc_1(x)
        x = self.lrelu(x)
        x = self.fc_2(x)
        self['out'] = self.lrelu(x)

        return self['out']



class ConvG(dock.Craft):
    def __init__(self, in_shape, latent_dim, scope='CONVG'):
        super(ConvG, self).__init__(scope)
        H, W, C = in_shape
        assert H % 4 == 0 and W % 4 == 0
        self.H = H
        self.W = W
        self.C = C

        self.lrelu = dock.LRelu()
        self.up = dock.Upscale(2)

        self.fc_1 = dock.Dense(latent_dim, 128 * (H // 4) * (W // 4))
        self.bn_1 = dock.BN(128, dim=2)
        pad = dock.autoPad((H // 2, W // 2), (3, 3))
        self.conv_1 = dock.Conv(128, 128, (3, 3), padding=pad)
        self.bn_2 = dock.BN(128, dim=2, mmnt=0.8)
        pad = dock.autoPad((H, W), (3, 3))
        self.conv_2 = dock.Conv(128, 64, (3, 3), padding=pad)
        self.bn_3 = dock.BN(64, dim=2, mmnt=0.8)
        pad = dock.autoPad((H, W), (3, 3))
        self.conv_3 = dock.Conv(64, C, (3, 3), padding=pad)

        self.tanh = dock.Tanh()
        self.reshape = dock.Reshape()

    def run(self, z):
        self['latent_code'] = z
        z = self.fc_1(self['latent_code'])
        z = self.reshape(z, (-1, 128, self.H // 4, self.W // 4))

        z = self.bn_1(z)
        z = self.up(z)
        z = self.conv_1(z)

        z = self.bn_2(z)
        z = self.lrelu(z)
        z = self.up(z)
        z = self.conv_2(z)

        z = self.bn_3(z)
        z = self.lrelu(z)
        z = self.conv_3(z)

        self['fake'] = self.tanh(z)

        return self['fake']

class DeconvG(dock.Craft):
    def __init__(self, in_shape, latent_dim, scope='DECONVG'):
        super(DeconvG, self).__init__(scope)
        H, W, C = in_shape
        assert H%4==0 and W%4==0
        self.H = H
        self.W = W
        self.C = C

        self.lrelu = dock.LRelu()

        self.fc_1 = dock.Dense(latent_dim, 128 * (H // 4) * (W // 4))
        self.bn_1 = dock.BN(128, dim=2)
        pad = dock.autoPad((H // 2, W // 2), (3, 3), stride=2)
        self.conv_1 = dock.TransConv(128, 128, (H // 2, W // 2), kernel=(3, 3), stride=2, padding=pad)
        self.bn_2 = dock.BN(128, dim=2, mmnt=0.8)
        pad = dock.autoPad((H, W), (3, 3), stride=2)
        self.conv_2 = dock.TransConv(128, 64, (H, W), kernel=(3, 3), stride=2, padding=pad)
        self.bn_3 = dock.BN(64, dim=2, mmnt=0.8)
        pad = dock.autoPad((H, W), (3, 3))
        self.conv_3 = dock.Conv(64, C, (3, 3), padding=pad)

        self.tanh = dock.Tanh()
        self.reshape = dock.Reshape()

    def run(self, z):
        self['latent_code'] = z
        z = self.fc_1(self['latent_code'])
        z = self.reshape(z, (-1, 128, self.H//4, self.W//4))

        z = self.bn_1(z)
        z = self.conv_1(z)

        z = self.bn_2(z)
        z = self.lrelu(z)
        z = self.conv_2(z)

        z = self.bn_3(z)
        z = self.lrelu(z)
        z = self.conv_3(z)

        self['fake'] = self.tanh(z)

        return self['fake']

class ConvD(dock.Craft):
    def __init__(self, in_shape, scope='CONVD'):
        super(ConvD, self).__init__(scope)
        H, W, C = in_shape
        self.H = H
        self.W = W
        self.C = C

        self.lrelu = dock.LRelu()
        self.flat = dock.Reshape()
        pad = dock.autoPad((H, W), (3, 3), stride=2)
        self.conv_1 = dock.Conv(C, 16, (3, 3), stride=2, padding=pad)
        pad = dock.autoPad((ceil(H / 2), ceil(W / 2)), (3, 3), stride=2)
        self.conv_2 = dock.Conv(16, 32, (3, 3), stride=2, padding=pad)
        self.bn_2 = dock.BN(32, dim=2, mmnt=0.8)
        pad = dock.autoPad((ceil(H / 4), ceil(W / 4)), (3, 3), stride=2)
        self.conv_3 = dock.Conv(32, 64, (3, 3), stride=2, padding=pad)
        self.bn_3 = dock.BN(64, dim=2, mmnt=0.8)
        pad = dock.autoPad((ceil(H / 8), ceil(W / 8)), (3, 3), stride=2)
        self.conv_4 = dock.Conv(64, 128, (3, 3), stride=2, padding=pad)
        self.bn_4 = dock.BN(128, dim=2, mmnt=0.8)

    def run(self, x):
        bs = x.shape[0]
        self['input'] = x
        x = self.conv_1(x)
        x = self.lrelu(x)

        x = self.conv_2(x)
        x = self.lrelu(x)
        x = self.bn_2(x)

        x = self.conv_3(x)
        x = self.lrelu(x)
        x = self.bn_3(x)

        x = self.conv_4(x)
        x = self.lrelu(x)
        x = self.bn_4(x)

        self['out'] = self.flat(x, (bs, -1))

        return self['out']



class ResBlock(dock.Craft):
    def __init__(self, in_shape, neck_chs, body_chs, stride=1, norm_fn=BN,
                 attention=False, spec_norm=False, w_init=dock.XavierNorm(), scope='RESBLK'):
        super(ResBlock, self).__init__(scope)
        H, W, C = in_shape
        self.norm_fn = norm_fn
        self.body_chs = body_chs
        self.cn = False
        if norm_fn == BN:
            self.n_1 = dock.BN(C, dim=2)
            self.n_2 = dock.BN(neck_chs, dim=2)
            # self.ds_n = dock.BN(self.body_chs, dim=2)
        elif norm_fn == IN:
            self.n_1 = dock.IN(C, dim=2)
            self.n_2 = dock.IN(neck_chs, dim=2)
            # self.ds_n = dock.IN(self.body_chs, dim=2)
        elif norm_fn == CBN:
            self.n_1 = dock.CBN(128, C, dim=2)
            self.n_2 = dock.CBN(128, neck_chs, dim=2)
            # self.ds_n = dock.CBN(128, self.body_chs, dim=2)
            self.cn = True
        elif norm_fn == CIN:
            self.n_1 = dock.CIN(128, C, dim=2)
            self.n_2 = dock.CIN(128, neck_chs, dim=2)
            # self.ds_n = dock.CIN(128, self.body_chs, dim=2)
            self.cn = True
        elif norm_fn == LN:
            self.n_1 = dock.LN(in_shape)
            self.n_2 = dock.LN((ceil(H / stride), ceil(W / stride), neck_chs))
            # self.ds_n = dock.LN((ceil(H / stride), ceil(W / stride), self.body_chs))
        elif norm_fn is None:
            self.n_1 = dock.Identity()
            self.n_2 = dock.Identity()
            # self.ds_n = dock.Identity()

        pad = dock.autoPad((H, W), (3, 3), stride=stride)
        if spec_norm:
            self.conv_1 = dock.SN(dock.Conv(C, neck_chs, (3, 3), stride=stride, padding=pad, w_init=w_init))
            self.conv_2 = dock.SN(dock.Conv(neck_chs, self.body_chs, (1, 1), w_init=w_init))
            self.ds_conv = dock.SN(dock.Conv(C, self.body_chs, (1, 1), stride, w_init=w_init))
        else:
            self.conv_1 = dock.Conv(C, neck_chs, (3, 3), stride=stride, padding=pad, w_init=w_init)
            self.conv_2 = dock.Conv(neck_chs, self.body_chs, (1, 1), w_init=w_init)
            self.ds_conv = dock.Conv(C, self.body_chs, (1, 1), stride, w_init=w_init)

        self.relu = dock.Relu()
        if attention and H==W==64:
            self.att = NonLocal((H // stride, W // stride, self.body_chs), self.body_chs//8, self.body_chs//2,
                                spec_norm, w_init)
        else:
            self.att = dock.Identity()

    def run(self, x):
        if self.cn:
            self['input'], self['noise'] = x
            y = self.n_1(self['input'], self['noise'])
            y = self.relu(y)
            y = self.conv_1(y)

            y = self.n_2(y, self['noise'])
            y = self.relu(y)
            y = self.conv_2(y)

            self['identity'] = self.ds_conv(self['input'])
        else:
            self['input'] = x
            y = self.n_1(self['input'])
            y = self.relu(y)
            y = self.conv_1(y)

            y = self.n_2(y)
            y = self.relu(y)
            y = self.conv_2(y)

            self['identity'] = self.ds_conv(self['input'])

        y += self['identity']
        y = self.att(y)

        return y


class ResG(dock.Craft):
    def __init__(self, in_shape, latent_dim, base_chs, norm_fn=BN, attention=False,
                 spec_norm=False, w_init=dock.XavierNorm(), scope='RESG'):
        super(ResG, self).__init__(scope)
        self.min_res = 4
        ichs = {128: [base_chs * c for c in (16, 16, 8, 4, 2)],
                256: [base_chs * c for c in (16, 16, 8, 8, 4, 2)]}
        ochs = {128: [base_chs * c for c in (16, 8, 4, 2, 1)],
                256: [base_chs * c for c in (16, 8, 8, 4, 2, 1)]}

        H, W, C = in_shape
        self.H = H
        self.W = W
        self.C = C
        min_size = min(H, W)
        self.ichs = ichs[min_size]
        self.ochs = ochs[min_size]

        self.up = dock.Upscale(2)
        self.relu = dock.Relu()
        pad = dock.autoPad((H, W), (3, 3))
        if spec_norm:
            self.conv = dock.SN(dock.Conv(self.ochs[-1], C, (3, 3), padding=pad, w_init=w_init))
            self.fc = dock.SN(dock.Dense(latent_dim, self.ichs[0] * self.min_res * self.min_res, w_init=w_init))
        else:
            self.conv = dock.Conv(self.ochs[-1], C, (3, 3), padding=pad, w_init=w_init)
            self.fc = dock.Dense(latent_dim, self.ichs[0] * self.min_res * self.min_res, w_init=w_init)

        for i in range(len(self.ochs)):
            setattr(self, 'blk_%d'%i, ResBlock((self.min_res * 2**i, self.min_res * 2**i, self.ichs[i]),
                                                self.ichs[i]//2, self.ochs[i], norm_fn=norm_fn,
                                                attention=attention, spec_norm=spec_norm, w_init=w_init))

        self.cn = False
        if norm_fn == BN:
            self.nf = dock.BN(self.ochs[-1], dim=2)
        elif norm_fn == IN:
            self.nf = dock.IN(self.ochs[-1], dim=2)
        elif norm_fn == CBN:
            self.nf = dock.CBN(latent_dim, self.ochs[-1], dim=2)
            self.cn = True
        elif norm_fn == CIN:
            self.nf = dock.CIN(latent_dim, self.ochs[-1], dim=2)
            self.cn = True
        elif norm_fn == LN:
            self.nf = dock.LN((H, W, self.ochs[-1]))
        elif norm_fn is None:
            self.nf = dock.Identity()

        self.tanh = dock.Tanh()
        self.reshape = dock.Reshape()

    def run(self, z):
        self['latent_code'] = z
        z = self.fc(self['latent_code'])
        z = self.reshape(z, (-1, self.ichs[0], self.min_res, self.min_res))

        if self.cn:
            for i in range(len(self.ochs)):
                rb = getattr(self, 'blk_%d' % i)
                z = rb((z, self['latent_code']))
                n_shape = z.shape
                n_shape[1] = 1
                noise = self.engine.coat(np.random.normal(size=n_shape).astype('float32'))
                strength = self.engine.coat(np.array(0).astype('float32'), as_const=False)
                z += strength * noise
                z = self.up(z)

            z = self.nf(z, self['latent_code'])
        else:
            for i in range(len(self.ochs)):
                rb = getattr(self, 'blk_%d' % i)
                z = rb(z)
                z = self.up(z)

            z = self.nf(z)

        z = self.relu(z)
        z = self.conv(z)

        self['fake'] = self.tanh(z)

        return self['fake']

class ResD(dock.Craft):
    def __init__(self, in_shape, base_chs, norm_fn=BN, attention=False,
                 spec_norm=False, w_init=dock.XavierNorm(), scope='RESD'):
        super(ResD, self).__init__(scope)
        H, W, C = in_shape
        min_size = min(H, W)
        ichs = {128: [base_chs * c for c in (1, 2, 4, 8)],
                256: [base_chs * c for c in (1, 2, 4, 8, 8)]}
        ochs = {128: [base_chs * c for c in (2, 4, 8, 16)],
                256: [base_chs * c for c in (2, 4, 8, 8, 16)]}
        self.ichs = ichs[min_size]
        self.ochs = ochs[min_size]

        self.relu = dock.Relu()
        self.flat = dock.Reshape()
        pad = dock.autoPad((H, W), (3, 3))
        if spec_norm:
            self.conv = dock.SN(dock.Conv(C, self.ichs[0], (3, 3), padding=pad))
        else:
            self.conv = dock.Conv(C, self.ichs[0], (3, 3), padding=pad)

        if norm_fn == CBN:
            norm_fn = BN
        if norm_fn == CIN:
            norm_fn = IN

        for i in range(len(self.ichs)):
            pad = dock.autoPad((H // 2**i, W // 2**i), (2, 2), stride=2)
            setattr(self, 'pool_%d'%i, dock.AvgPool((2, 2), padding=pad))
            setattr(self, 'blk_%d'%i,
                    ResBlock((H // 2**(i+1), W // 2**(i+1), self.ichs[i]), self.ichs[i]//2, self.ochs[i],
                              norm_fn=norm_fn, attention=attention, spec_norm=spec_norm, w_init=w_init))

        if norm_fn == BN:
            self.nf = dock.BN(self.ochs[-1], dim=2)
        elif norm_fn == IN:
            self.nf = dock.IN(self.ochs[-1], dim=2)
        elif norm_fn == LN:
            self.nf = dock.LN((H // 2**len(self.ichs), W // 2**len(self.ichs), self.ochs[-1]))
        elif norm_fn is None:
            self.nf = dock.Identity()

    def run(self, x):
        bs = x.shape[0]
        self['input'] = x
        x = self.conv(x)
        x = self.relu(x)

        for i in range(len(self.ichs)):
            pl = getattr(self, 'pool_%d'%i)
            rb = getattr(self, 'blk_%d'%i)
            x = pl(x)
            x = rb(x)

        x = self.nf(x)
        x = self.relu(x)

        self['out'] = self.flat(x, (bs, -1))

        return self['out']




class NonLocal(dock.Craft):
    def __init__(self, in_shape, att_chs, neck_chs, spec_norm, w_init, scope='NONLOCAL'):
        super(NonLocal, self).__init__(scope)
        # Channel multiplier
        H, W, C = in_shape
        self.H = H
        self.W = W
        self.C = C

        self.att_chs = att_chs
        self.neck_chs = neck_chs

        pad = dock.autoPad((H, W), (2, 2), 2)
        self.mpool_p = dock.MaxPool((2, 2), padding=pad)
        self.mpool_g = dock.MaxPool((2, 2), padding=pad)

        if spec_norm:
            self.theta = dock.SN(dock.Conv(self.C, self.att_chs, (1, 1), w_init=w_init))
            self.phi = dock.SN(dock.Conv(self.C, self.att_chs, (1, 1), w_init=w_init))
            self.g = dock.SN(dock.Conv(self.C, self.neck_chs, (1, 1), w_init=w_init))
            self.o = dock.SN(dock.Conv(self.neck_chs, self.C, (1, 1), w_init=w_init))
        else:
            self.theta = dock.Conv(self.C, self.att_chs, (1, 1), w_init=w_init)
            self.phi = dock.Conv(self.C, self.att_chs, (1, 1), w_init=w_init)
            self.g = dock.Conv(self.C, self.neck_chs, (1, 1), w_init=w_init)
            self.o = dock.Conv(self.neck_chs, self.C, (1, 1), w_init=w_init)

        self.flat = dock.Reshape()
        self.perm = dock.Permute()
        self.sftm = dock.Sftm()
        self.mm = dock.Dot()

    def forward(self, x):
        # Apply convs
        theta = self.theta(x)
        phi = self.phi(x)
        phi = self.mpool_p(phi)
        g = self.g(x)
        g = self.mpool_g(g)
        # Perform reshapes
        theta = self.flat(theta, (-1, self.att_chs, self.H * self.W))
        phi = self.flat(phi, (-1, self.att_chs, self.H * self.W // 4))
        g = self.flat(g, (-1, self.neck_chs, self.H * self.W // 4))
        # Matmul and softmax to get attention maps
        theta = self.perm(theta, (0, 2, 1))
        alpha = self.mm(theta, phi, in_batch=True)
        alpha = self.sftm(alpha)
        # Attention map times g path
        alpha = self.perm(alpha, (0, 2, 1))
        o = self.mm(g, alpha, in_batch=True)
        o = self.flat(o, (-1, self.neck_chs, self.H, self.W))
        o = self.o(o)
        return o + x





class BigResBlock(dock.Craft):
    def __init__(self, in_shape, neck_chs, body_chs, stride=1, norm_fn=BN, ups=None, downs=None,
                 attention=False, spec_norm=False, w_init=dock.XavierNorm(), scope='BIGRESBLK'):
        super(BigResBlock, self).__init__(scope)
        H, W, C = in_shape
        self.norm_fn = norm_fn
        self.body_chs = body_chs
        assert ups is None or downs is None
        self.ups = ups
        self.downs = downs
        self.cat = dock.Concat()

        self.cn = False
        if norm_fn == BN:
            self.n_1 = dock.BN(C, dim=2)
            self.n_2 = dock.BN(neck_chs, dim=2)
            # self.ds_n = dock.BN(self.body_chs, dim=2)
        elif norm_fn == IN:
            self.n_1 = dock.IN(C, dim=2)
            self.n_2 = dock.IN(neck_chs, dim=2)
            # self.ds_n = dock.IN(self.body_chs, dim=2)
        elif norm_fn == CBN:
            self.n_1 = dock.CBN(128, C, dim=2)
            self.n_2 = dock.CBN(128, neck_chs, dim=2)
            # self.ds_n = dock.CBN(128, self.body_chs, dim=2)
            self.cn = True
        elif norm_fn == CIN:
            self.n_1 = dock.CIN(128, C, dim=2)
            self.n_2 = dock.CIN(128, neck_chs, dim=2)
            # self.ds_n = dock.CIN(128, self.body_chs, dim=2)
            self.cn = True
        elif norm_fn == LN:
            self.n_1 = dock.LN(in_shape)
            self.n_2 = dock.LN((ceil(H / stride), ceil(W / stride), neck_chs))
            # self.ds_n = dock.LN((ceil(H / stride), ceil(W / stride), self.body_chs))
        elif norm_fn is None:
            self.n_1 = dock.Identity()
            self.n_2 = dock.Identity()
            # self.ds_n = dock.Identity()

        pad = dock.autoPad((H, W), (3, 3), stride=stride)
        if spec_norm:
            self.conv_1 = dock.SN(dock.Conv(C, neck_chs, (3, 3), stride=stride, padding=pad, w_init=w_init))
            self.conv_2 = dock.SN(dock.Conv(neck_chs, self.body_chs, (1, 1), w_init=w_init))
            self.ds_conv = dock.SN(dock.Conv(C, self.body_chs, (1, 1), stride, w_init=w_init))
        else:
            self.conv_1 = dock.Conv(C, neck_chs, (3, 3), stride=stride, padding=pad, w_init=w_init)
            self.conv_2 = dock.Conv(neck_chs, self.body_chs, (1, 1), w_init=w_init)
            if downs is None:
                self.rs_conv = dock.Conv(C, self.body_chs, (1, 1), stride, w_init=w_init)
            elif ups is None:
                self.rs_conv = dock.Conv(C, self.body_chs - C, (1, 1), stride, w_init=w_init)

        self.relu = dock.Relu()
        if attention and H == W == 64:
            self.att = NonLocal((H // stride, W // stride, self.body_chs), self.body_chs // 8, self.body_chs // 2,
                                spec_norm, w_init)
        else:
            self.att = dock.Identity()

    def shortcut(self, x):
        if self.downs is None:  # upscale
            return self.ups(self.rs_conv(x))
        elif self.ups is None:  # pooling
            x = self.downs(x)
            h = self.rs_conv(x)
            return self.cat((x, h), axis=1)

    def run(self, x):
        if self.cn:
            self['input'], self['noise'] = x
            y = self.n_1(self['input'], self['noise'])
            y = self.relu(y)
            if self.downs is None:  # upscale
                y = self.ups(y)
            y = self.conv_1(y)

            y = self.n_2(y, self['noise'])
            y = self.relu(y)
            if self.ups is None:  # pooling
                y = self.downs(y)
            y = self.conv_2(y)

            self['identity'] = self.shortcut(self['input'])
        else:
            self['input'] = x
            y = self.n_1(self['input'])
            y = self.relu(y)
            if self.downs is None:  # upscale
                y = self.ups(y)
            y = self.conv_1(y)

            y = self.n_2(y)
            y = self.relu(y)
            if self.ups is None:  # pooling
                y = self.downs(y)
            y = self.conv_2(y)

            self['identity'] = self.shortcut(self['input'])

        y += self['identity']
        y = self.att(y)

        return y


class BigG(dock.Craft):
    def __init__(self, in_shape, latent_dim, base_chs, norm_fn=BN, attention=False,
                 spec_norm=False, w_init=dock.XavierNorm(), scope='BIGG'):
        super(BigG, self).__init__(scope)
        self.min_res = 8
        ichs = {128: [base_chs * c for c in (16, 8, 4, 2)]}
        ochs = {128: [base_chs * c for c in (8, 4, 2, 1)]}

        H, W, C = in_shape
        self.H = H
        self.W = W
        self.C = C
        min_size = min(H, W)
        self.ichs = ichs[min_size]
        self.ochs = ochs[min_size]

        up = dock.Upscale(2)
        self.relu = dock.Relu()
        pad = dock.autoPad((H, W), (3, 3))
        if spec_norm:
            self.conv = dock.SN(dock.Conv(self.ochs[-1], C, (3, 3), padding=pad, w_init=w_init))
            self.fc = dock.SN(dock.Dense(latent_dim, self.ichs[0] * self.min_res * self.min_res, w_init=w_init))
        else:
            self.conv = dock.Conv(self.ochs[-1], C, (3, 3), padding=pad, w_init=w_init)
            self.fc = dock.Dense(latent_dim, self.ichs[0] * self.min_res * self.min_res, w_init=w_init)

        for i in range(len(self.ochs)):
            setattr(self, 'blk_%d' % i, BigResBlock((self.min_res * 2 ** i, self.min_res * 2 ** i, self.ichs[i]),
                                                     self.ichs[i] // 2, self.ochs[i], norm_fn=norm_fn, ups=up,
                                                     attention = attention, spec_norm = spec_norm, w_init = w_init))

            self.cn = False
            if norm_fn == BN:
                self.nf = dock.BN(self.ochs[-1], dim=2)
            elif norm_fn == IN:
                self.nf = dock.IN(self.ochs[-1], dim=2)
            elif norm_fn == CBN:
                self.nf = dock.CBN(latent_dim, self.ochs[-1], dim=2)
                self.cn = True
            elif norm_fn == CIN:
                self.nf = dock.CIN(latent_dim, self.ochs[-1], dim=2)
                self.cn = True
            elif norm_fn == LN:
                self.nf = dock.LN((H, W, self.ochs[-1]))
            elif norm_fn is None:
                self.nf = dock.Identity()

            self.tanh = dock.Tanh()
            self.reshape = dock.Reshape()

    def run(self, z):
        self['latent_code'] = z
        z = self.fc(self['latent_code'])
        z = self.reshape(z, (-1, self.ichs[0], self.min_res, self.min_res))

        if self.cn:
            for i in range(len(self.ochs)):
                rb = getattr(self, 'blk_%d' % i)
                z = rb((z, self['latent_code']))

            z = self.nf(z, self['latent_code'])
        else:
            for i in range(len(self.ochs)):
                rb = getattr(self, 'blk_%d' % i)
                z = rb(z)

            z = self.nf(z)

        z = self.relu(z)
        z = self.conv(z)

        self['fake'] = self.tanh(z)

        return self['fake']

class BigD(dock.Craft):
    def __init__(self, in_shape, base_chs, norm_fn=BN, attention=False,
                 spec_norm=False, w_init=dock.XavierNorm(), scope='BIGD'):
        super(BigD, self).__init__(scope)
        H, W, C = in_shape
        min_size = min(H, W)
        ichs = {128: [base_chs * c for c in (1, 2, 4, 8)]}
        ochs = {128: [base_chs * c for c in (2, 4, 8, 16)]}
        self.ichs = ichs[min_size]
        self.ochs = ochs[min_size]

        self.relu = dock.Relu()
        self.flat = dock.Reshape()
        pad = dock.autoPad((H, W), (3, 3))
        if spec_norm:
            self.conv = dock.SN(dock.Conv(C, self.ichs[0], (3, 3), padding=pad))
        else:
            self.conv = dock.Conv(C, self.ichs[0], (3, 3), padding=pad)

        if norm_fn == CBN:
            norm_fn = BN
        if norm_fn == CIN:
            norm_fn = IN

        for i in range(len(self.ichs)):
            pad = dock.autoPad((H // 2 ** i, W // 2 ** i), (2, 2), stride=2)
            #             setattr(self, 'pool_%d'%i, dock.AvgPool((2, 2), padding=pad))
            pool = dock.AvgPool((2, 2), padding=pad)
            setattr(self, 'blk_%d' % i,
                    BigResBlock((H // 2 ** (i + 1), W // 2 ** (i + 1), self.ichs[i]), self.ichs[i] // 2,
                                 self.ochs[i], norm_fn=norm_fn, downs=pool, attention=attention,
                                 spec_norm=spec_norm, w_init=w_init))

        if norm_fn == BN:
            self.nf = dock.BN(self.ochs[-1], dim=2)
        elif norm_fn == IN:
            self.nf = dock.IN(self.ochs[-1], dim=2)
        # elif norm_fn == CBN:
        #     self.nf = dock.CBN(128, self.ochs[-1], dim=2)
        #     self.cn = True
        # elif norm_fn == CIN:
        #     self.nf = dock.CIN(128, self.ochs[-1], dim=2)
        #     self.cn = True
        elif norm_fn == LN:
            self.nf = dock.LN((H // 2 ** len(self.ichs), W // 2 ** len(self.ichs), self.ochs[-1]))
        elif norm_fn is None:
            self.nf = dock.Identity()

    def run(self, x):
        bs = x.shape[0]
        self['input'] = x
        x = self.conv(x)
        x = self.relu(x)

        for i in range(len(self.ichs)):
            rb = getattr(self, 'blk_%d' % i)
            x = rb(x)

        x = self.nf(x)
        x = self.relu(x)

        self['out'] = self.flat(x, (bs, -1))

        return self['out']