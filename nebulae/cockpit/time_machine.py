#!/usr/bin/env python
'''
time_machine
Created by Seria at 23/12/2018 8:34 PM
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

def TimeMachine(config=None, ckpt_path=None, save_path=None, max_anchors=-1):
    rank = int(os.environ.get('RANK', -1))
    if config is None:
        param = {'ckpt_path': ckpt_path, 'save_path': save_path, 'max_anchors': max_anchors, 'rank': rank}
    else:
        config['ckpt_path'] = config.get('ckpt_path', ckpt_path)
        config['save_path'] = config.get('save_path', save_path)
        config['max_anchors'] = config.get('max_anchors', max_anchors)
        config['rank'] = config.get('rank', rank)
        param = config

    core = os.environ.get('NEB_CORE', 'PYTORCH')
    if core.upper() == 'TENSORFLOW':
        from .time_machine_tf import TimeMachineTF
        return TimeMachineTF(param)
    elif core.upper() == 'PYTORCH':
        from .time_machine_pt import TimeMachinePT
        return TimeMachinePT(param)
    else:
        raise ValueError('NEBULAE ERROR ⨷ %s is an unsupported core.' % core)