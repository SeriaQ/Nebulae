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
import tensorflow as tf
import os
from glob import glob

class TimeMachineTF(object):
    def __init__(self, param):
        '''
        Time Machine saves current states or restores saved states
        '''
        self.param = param
        self.counter = 1
        self.anchors = []

    def to(self, craft, optz=None, file='', ckpt_scope=None, frozen=False):
        assert self.param['ckpt_path'] is not None, Exception('NEBULAE ERROR ⨷ anchor location is not provided.')

        if not isinstance(craft, tf.keras.Model): # if wrapped in EMA
            craft = craft.hull

        ckpt_path = os.path.join(self.param['ckpt_path'], file)
        if os.path.isfile(ckpt_path):
            assert optz is None, Exception('NEBULAE ERROR ⨷ single checkpoint file does not contain optimizer states.')
            moment = os.path.dirname(ckpt_path)
            craft.load_weights(moment, by_name=True, skip_mismatch=not frozen)
        else:
            moment = tf.train.Checkpoint(net=craft, optimizer=optz.optz)
            ckpt_manager = tf.train.CheckpointManager(moment, ckpt_path, max_to_keep=None)
            moment.restore(ckpt_manager.latest_checkpoint)

        if self.param['rank'] <= 0:
            print('+' + ((10 + len(ckpt_path)) * '-') + '+')
            print('| Back to \033[1;34m%s\033[0m |' % ckpt_path)
            print('+' + ((10 + len(ckpt_path)) * '-') + '+')

    def drop(self, craft, optz=None, file='', save_scope=None, frozen=False):
        if self.param['rank']>0:
            return
        assert self.param['save_path'] is not None, Exception('NEBULAE ERROR ⨷ there is nowhere to drop anchor.')

        if not isinstance(craft, tf.keras.Model): # if wrapped in EMA
            craft = craft.hull

        save_path = os.path.join(self.param['save_path'], file)
        if save_path.endswith('.h5'):
            craft.save_weights(save_path)
        else:
            if optz is None:
                save_ckpt = tf.train.Checkpoint(net=craft)
            else:
                save_ckpt = tf.train.Checkpoint(net=craft, optimizer=optz.optz)
            max_anchors = self.param['max_anchors'] if self.param['max_anchors']>0 else None
            ckpt_manager = tf.train.CheckpointManager(save_ckpt, save_path, max_to_keep=max_anchors)
            ckpt_manager.save()
        print('| Anchor is dropped at \033[1;34m%s\033[0m |' % save_path)