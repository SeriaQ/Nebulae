#!/usr/bin/env python
'''
__init__
Created by Seria at 20/12/2018 4:54 PM
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
from .engine import Engine
from .time_machine import TimeMachine
import os

core = os.environ.get('NEB_CORE', 'PYTORCH')
if core.upper() == 'TENSORFLOW':
    from .multiverse_tf import Multiverse
elif core.upper() == 'PYTORCH':
    from .multiverse_pt import Multiverse
else:
    raise ValueError('NEBULAE ERROR ⨷ %s is an unsupported core.' % core)