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



class TimeMachinePT(object):
    def __init__(self, param):
        '''
        Time Machine saves current states or restores saved states
        '''
        self.param = param
        self.counter = 1
        self.anchors = []

    def to(self, craft, optz=None, file='', ckpt_scope=None, frozen=False):
        assert self.param['ckpt_path'] is not None, Exception('NEBULAE ERROR ⨷ anchor location is not provided.')

        ckpt_path = os.path.join(self.param['ckpt_path'], file)
        if os.path.isfile(ckpt_path):
            moment = ckpt_path
        else:
            architecture = glob(os.path.join(ckpt_path,'*.pth'))
            latest = -1
            moment = None
            for arch in architecture:
                last_mod = os.path.getmtime(arch)
                if last_mod > latest:
                    moment = arch
                    latest = last_mod
        if moment is None:
            raise Exception('NEBULAE ERROR ⨷ valid anchor is not found.')

        states = torch.load(moment)
        if optz is not None:
            optz.load_state_dict(states['optz'])
            states = states['net']
        if not ckpt_scope is None:
            ckpt_scope = ckpt_scope.replace('/', '.')
            states = {k: v for k, v in states.items() if k.startswith(ckpt_scope)}
        craft.load_state_dict(states, strict=frozen)
        if self.param['rank'] <= 0:
            print('+' + ((10 + len(moment)) * '-') + '+')
            print('| Back to \033[1;34m%s\033[0m |' % moment)
            print('+' + ((10 + len(moment)) * '-') + '+')

    def drop(self, craft, optz=None, file='', save_scope=None, frozen=False):
        if self.param['rank']>0:
            return
        assert self.param['save_path'] is not None, Exception('NEBULAE ERROR ⨷ there is nowhere to drop anchor.')

        save_path = os.path.join(self.param['save_path'], file)
        states = craft.state_dict()
        if save_scope is not None:
            save_scope = save_scope.replace('/', '.')
            states = {k:v for k,v in states.items() if k.startswith(save_scope)}
        if optz is not None:
            states = {'net': states, 'optz': optz.state_dict()}

        if save_path.endswith('.pth'):
            save_ckpt = save_path
        else:
            save_ckpt = os.path.join(save_path, '%s-%d.pth'%(craft.scope, self.counter))
        torch.save(states, save_ckpt)
        self.counter += 1
        self.anchors.append(save_ckpt)
        if self.param['max_anchors'] > 0 and len(self.anchors) > self.param['max_anchors']:
            os.remove(self.anchors[0])
            del self.anchors[0]
        print('| Anchor is dropped at \033[1;34m%s\033[0m |' % save_ckpt)