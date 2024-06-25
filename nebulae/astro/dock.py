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
from ..kit.utility import autopad

__all__ = ('coat', 'shell', 'autopad')

from .craft import *
from . import craft
__all__ += craft.__all__


def coat(datum, as_const=True, sync=True):
    raise NotImplementedError('NEBULAE ERROR ៙ coat function becomes valid only after setting up an Engine.')

def shell(datum, as_np=True, sync=False):
    raise NotImplementedError('NEBULAE ERROR ៙ shell function becomes valid only after setting up an Engine.')