#!/usr/bin/env python
'''
Created by Seria at 02/11/2018 3:38 PM
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

__all__ = ('Tank', 'Comburant')

core = os.environ.get('NEB_CORE', 'PYTORCH')
if core.upper() == 'TENSORFLOW':
    from .tank_tf import Tank
    from .comburant_tf import *
    from . import comburant_tf
    __all__ += comburant_tf.__all__
elif core.upper() == 'PYTORCH':
    from .tank_pt import Tank
    from .comburant_pt import *
    from . import comburant_pt
    __all__ += comburant_pt.__all__
else:
    raise ValueError('NEBULAE ERROR ⨷ %s is an unsupported core.' % core)