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
from ..kit.utility import autopad, cap

__all__ = ('coat', 'shell', 'autopad')

from .craft import *
from . import craft
__all__ += craft.__all__


def coat(datum, as_const=True, sync=True):
    raise NotImplementedError('NEBULAE ERROR ⨷ coat function becomes valid only after setting up an Engine.')

def shell(datum, as_np=True, sync=False):
    raise NotImplementedError('NEBULAE ERROR ⨷ shell function becomes valid only after setting up an Engine.')


class fn():
    def __init__(self, name, comp=None, symbol=''):
        self.name = name
        if comp is None: # atomic fn
            self.comp = []
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
                    if symbol==right_sym:
                        self.comp = [comp[0]] + comp[1].comp
                    else:
                        self.comp = [comp[0], comp[1]]
                elif right_sym=='':
                    if symbol == left_sym:
                        self.comp = comp[0].comp + [comp[1]]
                    else:
                        self.comp = [comp[0], comp[1]]
                else:
                    if symbol == left_sym:
                        self.comp = comp[0].comp + [comp[1]]
                    elif symbol == right_sym:
                        self.comp = [comp[0]] + comp[1].comp
                    else:
                        self.comp = comp
        self.symbol = symbol


    def __rshift__(self, other): # cascade
        return fn('', [self, other], '>')

    def __add__(self, other): # add
        return fn('', [self, other], '+')

    def __sub__(self, other): # sub
        return fn('', [self, other], '-')

    def __mul__(self, other): # multiply
        return fn('', [self, other], '*')

    def __matmul__(self, other): # dot
        return fn('', [self, other], '@')

    def __and__(self, other): # concat
        return fn('', [self, other], '&')

    def __or__(self, other):
        return fn('', [self, other], '|')

    def __xor__(self, other): # with
        return fn('', [self, other], '^')

    def _cap(self, scope):
        if '/' not in self.name:
            self.name = cap(self.name, scope)
        for c in self.comp:
            c._cap(scope)