#!/usr/bin/env python
'''
time_machine_tf
Created by Seria at 04/02/2019 4:35 PM
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
import torch
import os
from glob import glob
from ..rule import ENV_RANK



class TimeMachine(object):
    def __init__(self, ckpt_dir, save_dir, max_anchors=-1):
        '''
        Time Machine saves current states or restores saved states
        '''
        self.rank = int(os.environ.get(ENV_RANK, -1))
        self.ckpt_dir = ckpt_dir
        self.save_dir = save_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        self.max_anchors = max_anchors
        self.counter = 1
        self.anchors = []

    def to(self, craft, optz=None, file='', ckpt_scope=None, frozen=False):
        assert self.ckpt_dir is not None, Exception('NEBULAE ERROR ៙ anchor location is not provided.')

        ckpt_dir = os.path.join(self.ckpt_dir, file)
        if os.path.isfile(ckpt_dir):
            moment = ckpt_dir
        else:
            architecture = glob(os.path.join(ckpt_dir, '*.pth'))
            latest = -1
            moment = None
            for arch in architecture:
                last_mod = os.path.getmtime(arch)
                if last_mod > latest:
                    moment = arch
                    latest = last_mod
        if moment is None:
            raise Exception('NEBULAE ERROR ៙ valid anchor is not found.')

        states = torch.load(moment)
        if optz is not None:
            if isinstance(optz, dict):
                for k, v in optz.items():
                    v.load_state_dict(states[k])
            else:
                optz.load_state_dict(states['optz'])
            states = states['net']
        if not ckpt_scope is None:
            ckpt_scope = ckpt_scope.replace('/', '.')
            states = {k: v for k, v in states.items() if k.startswith(ckpt_scope)}
        craft.load_state_dict(states, strict=frozen)
        if self.rank <= 0:
            print('+' + ((10 + len(moment)) * '-') + '+')
            print('| Back to \033[1;34m%s\033[0m |' % moment)
            print('+' + ((10 + len(moment)) * '-') + '+')

    def drop(self, craft, optz=None, file='', save_scope=None, frozen=False):
        if self.rank>0:
            return
        assert self.save_dir is not None, Exception('NEBULAE ERROR ៙ there is nowhere to drop anchor.')

        save_dir = os.path.join(self.save_dir, file)
        states = craft.state_dict()
        if save_scope is not None:
            save_scope = save_scope.replace('/', '.')
            states = {k:v for k,v in states.items() if k.startswith(save_scope)}
        if optz is not None:
            if isinstance(optz, dict):
                states = {'net': states}
                for k, v in optz.items():
                    states[k] = v.state_dict()
            else:
                states = {'net': states, 'optz': optz.state_dict()}

        if save_dir.endswith('.pth'):
            save_ckpt = save_dir
        else:
            save_name = getattr(craft, 'scope', craft.__class__.__name__)
            save_ckpt = os.path.join(save_dir, '%s-%d.pth'%(save_name, self.counter))
        torch.save(states, save_ckpt)
        self.counter += 1
        self.anchors.append(save_ckpt)
        if self.max_anchors > 0 and len(self.anchors) > self.max_anchors:
            os.remove(self.anchors[0])
            del self.anchors[0]
        print('| Anchor is dropped at \033[1;34m%s\033[0m |' % save_ckpt)