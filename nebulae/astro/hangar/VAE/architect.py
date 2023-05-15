#!/usr/bin/env python
'''
architect
Created by Seria at 2022/11/7 12:08 PM
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
import numpy as np
from math import ceil


BN = 20
CBN = 21
IN = 22
CIN = 23
LN = 24



class MLPE(dock.Craft):
    def __init__(self, in_shape, hidden_dim, latent_dim, scope='MLPE'):
        super(MLPE, self).__init__(scope)
        H, W, C = in_shape
        in_dim = H * W * C
        self.in_dim = in_dim
        self.rsp = dock.Reshape()
        self.hidden = dock.Dense(in_dim, hidden_dim)
        self.relu = dock.Relu()
        self.mu = dock.Dense(hidden_dim, latent_dim)
        self.log_var = dock.Dense(hidden_dim, latent_dim)
        self.exp = dock.Exp()

    def reparam(self, mu, lv):
        eps = dock.coat(np.random.normal(size=mu.shape).astype(np.float32))
        return mu + self.exp(lv / 2) * eps

    def run(self, x):
        x = self.rsp(x, (-1, self.in_dim))
        x = self.hidden(x)
        x = self.relu(x)
        mu = self.mu(x)
        lv = self.log_var(x)
        z = self.reparam(mu, lv)
        return mu, lv, z


class MLPD(dock.Craft):
    def __init__(self, out_shape, hidden_dim, latent_dim, scope='MLPD'):
        super(MLPD, self).__init__(scope)
        H, W, C = out_shape
        out_dim = H * W * C
        self.out_shape = tuple(out_shape[-1:] + out_shape[:2])
        self.hidden = dock.Dense(latent_dim, hidden_dim)
        self.relu = dock.Relu()
        self.fc = dock.Dense(hidden_dim, out_dim)
        self.tanh = dock.Tanh()
        self.rsp = dock.Reshape()

    def run(self, z):
        z = self.hidden(z)
        z = self.relu(z)
        z = self.fc(z)
        z = self.tanh(z)
        x = self.rsp(z, (-1,) + self.out_shape)
        return x




class DeconvD(dock.Craft):
    def __init__(self, out_shape, latent_dim, scope='DECONVD'):
        super(DeconvD, self).__init__(scope)
        H, W, C = out_shape
        assert H%4==0 and W%4==0
        self.H = H
        self.W = W
        self.C = C

        self.lrelu = dock.LRelu()

        self.fc_1 = dock.Dense(latent_dim, 128 * (H // 4) * (W // 4))
        self.bn_1 = dock.BN(128, dim=2)
        pad = dock.autopad((3, 3), stride=2, size=(H // 2, W // 2))
        self.conv_1 = dock.TransConv(128, 128, (H // 2, W // 2), kernel=(3, 3), stride=2, padding=pad)
        self.bn_2 = dock.BN(128, dim=2, mmnt=0.8)
        pad = dock.autopad((3, 3), stride=2, size=(H, W))
        self.conv_2 = dock.TransConv(128, 64, (H, W), kernel=(3, 3), stride=2, padding=pad)
        self.bn_3 = dock.BN(64, dim=2, mmnt=0.8)
        pad = dock.autopad((3, 3), size=(H, W))
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


class ConvE(dock.Craft):
    def __init__(self, in_shape, scope='CONVE'):
        super(ConvE, self).__init__(scope)
        H, W, C = in_shape
        self.H = H
        self.W = W
        self.C = C

        self.lrelu = dock.LRelu()
        self.flat = dock.Reshape()
        self.exp = dock.Exp()
        pad = dock.autopad((3, 3), stride=2, size=(H, W))
        self.conv_1 = dock.Conv(C, 16, (3, 3), stride=2, padding=pad)
        pad = dock.autopad((3, 3), stride=2, size=(ceil(H / 2), ceil(W / 2)))
        self.conv_2 = dock.Conv(16, 32, (3, 3), stride=2, padding=pad)
        self.bn_2 = dock.BN(32, dim=2, mmnt=0.8)
        pad = dock.autopad((3, 3), stride=2, size=(ceil(H / 4), ceil(W / 4)))
        self.conv_3 = dock.Conv(32, 64, (3, 3), stride=2, padding=pad)
        self.bn_3 = dock.BN(64, dim=2, mmnt=0.8)
        pad = dock.autopad((3, 3), stride=2, size=(ceil(H / 8), ceil(W / 8)))
        self.conv_m = dock.Conv(64, 128, (3, 3), stride=2, padding=pad)
        self.bn_m = dock.BN(128, dim=2, mmnt=0.8)
        pad = dock.autopad((3, 3), stride=2, size=(ceil(H / 8), ceil(W / 8)))
        self.conv_v = dock.Conv(64, 128, (3, 3), stride=2, padding=pad)
        self.bn_v = dock.BN(128, dim=2, mmnt=0.8)

    def reparam(self, mu, lv):
        eps = dock.coat(np.random.normal(size=mu.shape).astype(np.float32))
        return mu + self.exp(lv / 2) * eps

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

        mu = self.conv_m(x)
        mu = self.lrelu(mu)
        mu = self.bn_m(mu)
        mu = self.flat(mu, (bs, -1))

        lv = self.conv_v(x)
        lv = self.lrelu(lv)
        lv = self.bn_v(lv)
        lv = self.flat(lv, (bs, -1))

        z = self.reparam(mu, lv)

        return mu, lv, z









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

        pad = dock.autopad((3, 3), stride=stride, size=(H, W))
        if spec_norm:
            self.conv_1 = dock.SN(dock.Conv(C, neck_chs, (3, 3), stride=stride, padding=pad, w_init=w_init))
            self.conv_2 = dock.SN(dock.Conv(neck_chs, self.body_chs, (1, 1), w_init=w_init))
            self.ds_conv = dock.SN(dock.Conv(C, self.body_chs, (1, 1), stride, w_init=w_init))
        else:
            self.conv_1 = dock.Conv(C, neck_chs, (3, 3), stride=stride, padding=pad, w_init=w_init)
            self.conv_2 = dock.Conv(neck_chs, self.body_chs, (1, 1), w_init=w_init)
            self.ds_conv = dock.Conv(C, self.body_chs, (1, 1), stride, w_init=w_init)

        self.relu = dock.Relu()
        # if attention and H==W==64:
        #     self.att = NonLocal((H // stride, W // stride, self.body_chs), self.body_chs//8, self.body_chs//2,
        #                         spec_norm, w_init)
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



class ResD(dock.Craft):
    def __init__(self, in_shape, base_chs, latent_dim, norm_fn=BN, attention=False,
                 spec_norm=False, w_init=dock.XavierNorm(), scope='RESD'):
        super(ResD, self).__init__(scope)
        self.min_res = 4
        ichs = {32: [base_chs * c for c in (4, 4, 2)],
                64: [base_chs * c for c in (8, 8, 4, 2)],
                128: [base_chs * c for c in (16, 16, 8, 4, 2)],
                256: [base_chs * c for c in (16, 16, 8, 8, 4, 2)]}
        ochs = {32: [base_chs * c for c in (4, 2, 1)],
                64: [base_chs * c for c in (8, 4, 2, 1)],
                128: [base_chs * c for c in (16, 8, 4, 2, 1)],
                256: [base_chs * c for c in (16, 8, 8, 4, 2, 1)]}

        H, W, C = in_shape
        self.H = H
        self.W = W
        self.C = C
        min_size = min(H, W)
        self.ichs = ichs[min_size]
        self.ochs = ochs[min_size]

        self.up = dock.Zoom(scale=(2, 2))
        self.relu = dock.Relu()
        pad = dock.autopad((3, 3), size=(H, W))
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
                noise = dock.coat(np.random.normal(size=n_shape).astype('float32'))
                strength = dock.coat(np.array(0).astype('float32'), as_const=False)
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

class ResE(dock.Craft):
    def __init__(self, in_shape, base_chs, latent_dim, norm_fn=BN, attention=False,
                 spec_norm=False, w_init=dock.XavierNorm(), scope='RESE'):
        super(ResE, self).__init__(scope)
        H, W, C = in_shape
        min_size = min(H, W)
        ichs = {32: [base_chs * c for c in (1, 2, 4)],
                64: [base_chs * c for c in (1, 2, 4, 4)],
                128: [base_chs * c for c in (1, 2, 4, 8)],
                256: [base_chs * c for c in (1, 2, 4, 8, 8)]}
        ochs = {32: [base_chs * c for c in (2, 4, 4)],
                64: [base_chs * c for c in (2, 4, 4, 8)],
                128: [base_chs * c for c in (2, 4, 8, 16)],
                256: [base_chs * c for c in (2, 4, 8, 8, 16)]}
        factor = {32: base_chs / 16,
                  64: base_chs / 32,
                  128: base_chs / 16,
                  256: base_chs / 64}
        self.ichs = ichs[min_size]
        self.ochs = ochs[min_size]

        self.relu = dock.Relu()
        self.flat = dock.Reshape()
        pad = dock.autopad((3, 3), size=(H, W))
        if spec_norm:
            self.conv = dock.SN(dock.Conv(C, self.ichs[0], (3, 3), padding=pad))
        else:
            self.conv = dock.Conv(C, self.ichs[0], (3, 3), padding=pad)

        if norm_fn == CBN:
            norm_fn = BN
        if norm_fn == CIN:
            norm_fn = IN

        for i in range(len(self.ichs)):
            pad = dock.autopad((2, 2), stride=2, size=(H // 2**i, W // 2**i))
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

        hidden_dim = int(H * W * factor[min_size])
        self.mu = dock.Dense(hidden_dim, latent_dim)
        self.log_var = dock.Dense(hidden_dim, latent_dim)
        self.exp = dock.Exp()

    def reparam(self, mu, lv):
        eps = dock.coat(np.random.normal(size=mu.shape).astype(np.float32))
        return mu + self.exp(lv / 2) * eps

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

        self['hidden'] = self.flat(x, (bs, -1))
        mu = self.mu(self['hidden'])
        lv = self.log_var(self['hidden'])
        z = self.reparam(mu, lv)
        return mu, lv, z



class VecQuant(dock.Craft):
    def __init__(self, k, d, scope='VECQUANT'):
        super(VecQuant, self).__init__(scope)
        self.dim = d
        self.embd = dock.Embed(k, d)
        self.resp = dock.Reshape()
        self.perm = dock.Permute()
        self.sum = dock.Sum()
        self.dot = dock.Dot()
        self.expd = dock.Expand()
        self.tile = dock.Tile()
        self.stk = dock.Stack()
        self.amin = dock.Argmin()

    def run(self, z):
        # z: B x D x M x M
        # c: K x D
        codebook = self.embd.weights()
        z_square = self.expd(self.sum(z ** 2, 1), -1) # B x M x M x 1
        c_square = self.tile(self.sum(codebook ** 2, 1), (1, 1, 1, 1)) # 1 x 1 x 1 x K
        in_prod = self.dot(z, codebook, ([1], [-1])) # B x M x M x K
        d_square = z_square + c_square - 2*in_prod # B x M x M x K
        B, M, N = d_square.shape[:3]
        idx_code = self.resp(self.amin(d_square, -1), (-1,))
        q = self.perm(self.resp(codebook[idx_code], (B, M, N, -1)), (0, 3, 1, 2))
        return q



class ResVQD(dock.Craft):
    def __init__(self, in_shape, base_chs, norm_fn=BN, attention=False,
                 spec_norm=False, w_init=dock.XavierNorm(), scope='RESVQD'):
        super(ResVQD, self).__init__(scope)
        self.min_res = 4
        ichs = {32: [base_chs * c for c in (4, 4, 2)],
                64: [base_chs * c for c in (8, 8, 4, 2)],
                128: [base_chs * c for c in (16, 16, 8, 4, 2)],
                256: [base_chs * c for c in (16, 16, 8, 8, 4, 2)]}
        ochs = {32: [base_chs * c for c in (4, 2, 1)],
                64: [base_chs * c for c in (8, 4, 2, 1)],
                128: [base_chs * c for c in (16, 8, 4, 2, 1)],
                256: [base_chs * c for c in (16, 8, 8, 4, 2, 1)]}

        H, W, C = in_shape
        self.H = H
        self.W = W
        self.C = C
        min_size = min(H, W)
        self.ichs = ichs[min_size]
        self.ochs = ochs[min_size]

        self.up = dock.Zoom(scale=(2, 2))
        self.relu = dock.Relu()
        pad = dock.autopad((3, 3), size=(H, W))
        if spec_norm:
            self.conv = dock.SN(dock.Conv(self.ochs[-1], C, (3, 3), padding=pad, w_init=w_init))
        else:
            self.conv = dock.Conv(self.ochs[-1], C, (3, 3), padding=pad, w_init=w_init)

        for i in range(len(self.ochs)):
            setattr(self, 'blk_%d'%i, ResBlock((self.min_res * 2**i, self.min_res * 2**i, self.ichs[i]),
                                                self.ichs[i]//2, self.ochs[i], norm_fn=norm_fn,
                                                attention=attention, spec_norm=spec_norm, w_init=w_init))

        self.cn = False
        if norm_fn == BN:
            self.nf = dock.BN(self.ochs[-1], dim=2)
        elif norm_fn == IN:
            self.nf = dock.IN(self.ochs[-1], dim=2)
        # elif norm_fn == CBN:
        #     self.nf = dock.CBN(latent_dim, self.ochs[-1], dim=2)
        #     self.cn = True
        # elif norm_fn == CIN:
        #     self.nf = dock.CIN(latent_dim, self.ochs[-1], dim=2)
        #     self.cn = True
        elif norm_fn == LN:
            self.nf = dock.LN((H, W, self.ochs[-1]))
        elif norm_fn is None:
            self.nf = dock.Identity()

        self.tanh = dock.Tanh()
        self.reshape = dock.Reshape()

    def run(self, z):
        for i in range(len(self.ochs)):
            rb = getattr(self, 'blk_%d' % i)
            z = rb(z)
            z = self.up(z)

        z = self.nf(z)

        z = self.relu(z)
        z = self.conv(z)

        self['fake'] = self.tanh(z)

        return self['fake']



class ResVQE(dock.Craft):
    def __init__(self, in_shape, base_chs, ncodes, norm_fn=BN, attention=False,
                 spec_norm=False, w_init=dock.XavierNorm(), scope='RESVQE'):
        super(ResVQE, self).__init__(scope)
        H, W, C = in_shape
        min_size = min(H, W)
        ichs = {32: [base_chs * c for c in (1, 2, 4)],
                64: [base_chs * c for c in (1, 2, 4, 4)],
                128: [base_chs * c for c in (1, 2, 4, 8)],
                256: [base_chs * c for c in (1, 2, 4, 8, 8)]}
        ochs = {32: [base_chs * c for c in (2, 4, 4)],
                64: [base_chs * c for c in (2, 4, 4, 8)],
                128: [base_chs * c for c in (2, 4, 8, 16)],
                256: [base_chs * c for c in (2, 4, 8, 8, 16)]}
        factor = {32: base_chs / 16,
                  64: base_chs / 32,
                  128: base_chs / 16,
                  256: base_chs / 64}
        self.ichs = ichs[min_size]
        self.ochs = ochs[min_size]

        self.relu = dock.Relu()
        self.tanh = dock.Tanh()
        self.flat = dock.Reshape()
        pad = dock.autopad((3, 3), size=(H, W))
        if spec_norm:
            self.conv = dock.SN(dock.Conv(C, self.ichs[0], (3, 3), padding=pad))
        else:
            self.conv = dock.Conv(C, self.ichs[0], (3, 3), padding=pad)

        if norm_fn == CBN:
            norm_fn = BN
        if norm_fn == CIN:
            norm_fn = IN

        for i in range(len(self.ichs)):
            pad = dock.autopad((2, 2), stride=2, size=(H // 2**i, W // 2**i))
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
        self['input'] = x
        x = self.conv(x)
        x = self.relu(x)

        for i in range(len(self.ichs)):
            pl = getattr(self, 'pool_%d'%i)
            rb = getattr(self, 'blk_%d'%i)
            x = pl(x)
            x = rb(x)

        x = self.nf(x)
        x = self.tanh(x)

        return x