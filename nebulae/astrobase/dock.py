#!/usr/bin/env python
'''
component
Created by Seria at 25/11/2018 2:58 PM
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
import os
from ..toolkit.utility import cap, autoPad
from collections import namedtuple

__all__ = ('Craft', 'Pod', 'Tensor', 'coat', 'shell')


Tensor = namedtuple('tensor', ('key', 'val'))
coat = None
shell = None


class Pod():
    def __init__(self, name, comp=[], symbol=''):
        self.name = name
        if len(comp)==0: # atomic pod
            self.comp = comp
        else:
            assert len(comp)==2
            left_sym = comp[0].symbol
            right_sym = comp[1].symbol
            if left_sym == right_sym:
                if right_sym=='':
                    self.comp = comp
                else:
                    self.comp = comp[0].comp + comp[1].comp
            else:
                if left_sym=='':
                    self.comp = [comp[0]] + comp[1].comp
                elif right_sym=='':
                    self.comp = comp[0].comp + [comp[1]]
                else:
                    self.comp = comp
        self.symbol = symbol


    def __gt__(self, other):
        return Pod('', [self, other], '>')

    def __add__(self, other):
        return Pod('', [self, other], '+')

    def __sub__(self, other):
        return Pod('', [self, other], '-')

    def __mul__(self, other):
        return Pod('', [self, other], '*')

    def __matmul__(self, other):
        return Pod('', [self, other], '@')

    def __and__(self, other):
        return Pod('', [self, other], '&')

    def __or__(self, other):
        return Pod('', [self, other], '|')

    def __xor__(self, other):
        return Pod('', [self, other], '^')

    def _cap(self, scope):
        if '/' not in self.name:
            self.name = cap(self.name, scope)
        for c in self.comp:
            c._cap(scope)


core = os.environ.get('NEB_CORE', 'PYTORCH')
if core.upper() == 'TENSORFLOW':
    from .craft_tf import *
    from . import craft_tf
    __all__ += craft_tf.__all__
elif core.upper() == 'PYTORCH':
    from .craft_pt import *
    from . import craft_pt
    __all__ += craft_pt.__all__
else:
    raise ValueError('NEBULAE ERROR ⨷ %s is an unsupported core.' % core)