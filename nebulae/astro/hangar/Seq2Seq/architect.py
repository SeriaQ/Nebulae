#!/usr/bin/env python
'''
architect
Created by Seria at 2021/5/22 11:23 AM
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


class RNNE(dock.Craft):
    def __init__(self, in_chs, hid_chs, nlayers, voc_size, scope='RNNE'):
        super(RNNE, self).__init__(scope)
        self.emb = dock.Embed(voc_size, in_chs)
        self.rnn = dock.RNN(in_chs, hid_chs, nlayers)

    def run(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        return y, h


class BiRNNE(dock.Craft):
    def __init__(self, in_chs, hid_chs, nlayers, voc_size, scope='BIRNNE'):
        super(BiRNNE, self).__init__(scope)
        self.emb = dock.Embed(voc_size, in_chs)
        self.birnn = dock.BiRNN(in_chs, hid_chs, nlayers)

    def run(self, x, h=None):
        x = self.emb(x)
        y, h = self.birnn(x, h)
        return y, h


class RNND(dock.Craft):
    def __init__(self, in_chs, hid_chs, nlayers, voc_size, scope='RNND'):
        super(RNND, self).__init__(scope)
        self.emb = dock.Embed(voc_size, in_chs)
        self.relu = dock.Relu()
        self.rnn = dock.RNN(in_chs, hid_chs, nlayers)
        self.fc = dock.Dense(hid_chs, voc_size)

    def run(self, x, h=None):
        x = self.emb(x)
        x = self.relu(x)
        y, h = self.rnn(x, h)
        y = self.fc(y)
        return y, h


# class AttnRNND(dock.Craft):
#     def __init__(self, in_chs, hid_chs, nlayers, voc_size, max_len, scope='ATTNRNND'):
#         super(AttnRNND, self).__init__(scope)
#         self.emb = dock.Embed(voc_size, in_chs)
#         self.lut = [p for p in self.emb.vars()]
#         self.relu = dock.Relu()
#         self.rnn = dock.RNN(in_chs, hid_chs, nlayers)
#         self.fc = dock.Dense(hid_chs, voc_size)
#         self.drp = dock.Dropout(0.1, dim=1)
#         self.perm = dock.Permute()
#         # self.sqsh = dock.Squash()
#         # self.expd = dock.Expand()
#
#         self.attn = dock.Dense(in_chs+hid_chs, max_len)
#         self.proj = dock.Dense(hid_chs+hid_chs, hid_chs)
#         self.dot = dock.Dot()
#         self.cat = dock.Concat()
#         self.sftm = dock.Sftm()
#
#     def run(self, x, o, h=None):
#         x = self.emb(x)
#         x = self.drp(x)
#
#         a = self.cat((x, h))
#         a = self.attn(a)
#         a = self.sftm(a)
#         _a = self.perm(a, (1, 0, 2))
#         _o = self.perm(o, (1, 0, 2))
#         a = self.dot(_a, _o, in_batch=True)
#         a = self.perm(a, (1, 0, 2))
#         h = self.cat((h, a))
#         h = self.proj(h)
#
#         x = self.relu(x)
#         y, h = self.rnn(x, h)
#         y = self.fc(y)
#         # y = self.expd(self.dot(self.sqsh(y, 0), self.perm(self.lut[0], (1,0))), 0)
#         return y, h


class AttnRNND(dock.Craft):
    def __init__(self, in_chs, hid_chs, nlayers, voc_size, max_len, scope='ATTNRNND'):
        super(AttnRNND, self).__init__(scope)
        self.nlay = nlayers
        self.hchs = hid_chs

        self.emb = dock.Embed(voc_size, in_chs)
        self.resh = dock.Reshape()
        self.perm = dock.Permute()
        self.expd = dock.Expand()
        self.sqsh = dock.Squash()
        self.relu = dock.Relu()
        self.rnn = dock.RNN(in_chs+hid_chs, hid_chs, nlayers)
        self.fc = dock.Dense(hid_chs, voc_size)
        self.do = dock.Dense(hid_chs, hid_chs)
        self.dh = dock.Dense(hid_chs, hid_chs)
        self.da = dock.Dense(hid_chs, 1)

        # self.proj = dock.Dense(in_chs+2*hid_chs, in_chs)
        self.dot = dock.Dot()
        self.cat = dock.Concat()
        self.sftm = dock.Sftm()

    def run(self, x, o, h=None):
        x = self.emb(x)
        h = self.resh(h, (self.nlay, 1, -1))

        _o = self.perm(self.do(o), (1, 0, 2)) # B x L x H*NDIR
        _h = self.perm(self.dh(h), (1, 2, 0)) # B x H*NDIR x NLAY
        # a = self.dot(_o, _h, in_batch=True) # B x L x NLAY
        a = _o + self.expd(self.sqsh(_h, -1), 1) # B x L x H*NDIR
        a = self.expd(self.sqsh(self.da(a), -1), 1) # B x 1 x L
        a = self.sftm(a)
        a = self.dot(a, _o, in_batch=True) # B x 1 x H*NDIR
        a = self.perm(a, (1, 0, 2))
        x = self.cat((x, a)) # 1 x B x 2*H*NDIR
        # x = self.proj(x)

        # x = self.relu(x)
        y, h = self.rnn(x, h)
        y = self.fc(y)
        return y, h


class LSTME(dock.Craft):
    def __init__(self, in_chs, hid_chs, nlayers, voc_size, scope='LSTME'):
        super(LSTME, self).__init__(scope)
        self.emb = dock.Embed(voc_size, in_chs)
        self.lstm = dock.LSTM(in_chs, hid_chs, nlayers)

    def run(self, x, h=None, c=None):
        x = self.emb(x)
        y, h, c = self.lstm(x, h, c)
        return y, h, c


class LSTMD(dock.Craft):
    def __init__(self, in_chs, hid_chs, nlayers, voc_size, attention=0, scope='LSTMD'):
        super(LSTMD, self).__init__(scope)
        self.emb = dock.Embed(voc_size, in_chs)
        self.relu = dock.Relu()
        self.lstm = dock.LSTM(in_chs, hid_chs, nlayers)
        self.fc = dock.Dense(hid_chs, voc_size)
        if attention>0:
            self.attn = dock.Dense(in_chs+hid_chs, attention)

    def run(self, x, h=None, c=None):
        x = self.emb(x)
        x = self.relu(x)
        y, h, c = self.lstm(x, h, c)
        y = self.fc(y)
        return y, h, c


class AttnLSTMD(dock.Craft):
    def __init__(self, in_chs, hid_chs, nlayers, voc_size, max_len, scope='ATTNLSTMD'):
        super(AttnLSTMD, self).__init__(scope)
        self.emb = dock.Embed(voc_size, in_chs)
        self.relu = dock.Relu()
        self.rnn = dock.LSTM(in_chs, hid_chs, nlayers)
        self.fc = dock.Dense(hid_chs, voc_size)
        self.perm = dock.Permute()

        self.attn = dock.Dense(in_chs+hid_chs, max_len)
        self.proj = dock.Dense(hid_chs+hid_chs, hid_chs)
        self.dot = dock.Dot()
        self.cat = dock.Concat()
        self.sftm = dock.Sftm()

    def run(self, x, o, h=None, c=None):
        x = self.emb(x)

        a = self.cat((x, c))
        a = self.attn(a)
        a = self.sftm(a)
        _a = self.perm(a, (1, 0, 2))
        _o = self.perm(o, (1, 0, 2))
        a = self.dot(_a, _o, in_batch=True)
        a = self.perm(a, (1, 0, 2))
        c = self.cat((c, a))
        c = self.proj(c)

        x = self.relu(x)
        y, h, c = self.rnn(x, h, c)
        y = self.fc(y)
        return y, h, c