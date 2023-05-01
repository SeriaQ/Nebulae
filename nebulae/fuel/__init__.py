#!/usr/bin/env python
'''
Created by Seria at 02/11/2018 3:39 PM
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
from .tank import *
from .comburant import *
from .generator import *
from . import tank, comburant, generator

from ..kit.utility import _merge_fuel as merge
from ..kit.utility import _fill_fuel as fill
from ..kit.utility import _deduct_fuel as deduct

__all__ = ('merge', 'fill', 'deduct') + comburant.__all__ + generator.__all__ + tank.__all__