#!/usr/bin/env python
'''
engine
Created by Seria at 23/11/2018 2:36 PM
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
from ..law import Constant

def Engine(config=None, device=None, ngpus=1, least_mem=2048, avail_gpus=(), multi_piston=False):
    rank = int(os.environ.get('RANK', -1))
    ngpus = int(os.environ.get('WORLD_SIZE', ngpus))
    if config is None:
        param = {'device': device, 'ngpus': ngpus, 'least_mem': least_mem, 'avail_gpus': avail_gpus,
                 'multi_piston': multi_piston, 'rank': rank}
    else:
        config['ngpus'] = config.get('ngpus', ngpus)
        config['least_mem'] = config.get('least_mem', least_mem)
        config['avail_gpus'] = config.get('avail_gpus', avail_gpus)
        config['multi_piston'] = config.get('multi_piston', multi_piston)
        config['rank'] = config.get('rank', rank)
        param = config
    if not isinstance(param['ngpus'], int):
        raise TypeError('NEBULAE ERROR ⨷ number of gpus must be an integer.')

    core = os.environ.get('NEB_CORE', 'PYTORCH')
    if core.upper() == 'TENSORFLOW':
        from .engine_tf import EngineTF
        return EngineTF(param)
    elif core.upper() == 'PYTORCH':
        from .engine_pt import EnginePT
        return EnginePT(param)
    else:
        raise ValueError('NEBULAE ERROR ⨷ %s is an unsupported core.' % core)