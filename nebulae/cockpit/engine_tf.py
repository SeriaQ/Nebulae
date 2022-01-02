#!/usr/bin/env python
'''
engine_tf
Created by Seria at 04/02/2019 4:31 PM
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
import tensorflow as tf
from tensorflow.python.distribute.values import PerReplica
from ..toolkit.utility import GPUtil

class EngineTF(object):
    '''
    Param:
    device: 'gpu' or 'cpu'
    available_gpus:
    least_mem
    '''
    def __init__(self, param):
        self.param = param
        # look for available gpu devices
        if self.param['device'].lower() == 'gpu':
            if len(self.param['avail_gpus'])==0:
                gputil = GPUtil()
                gpus = gputil.available(self.param['ngpus'], self.param['least_mem'])
                if len(gpus) == 0:
                    raise Exception('NEBULAE ERROR ⨷ no enough available gpu', gpus)
                # TODO: multi-gpu training is to be supported
            else:
                gpus = self.param['avail_gpus']
            # convert gpu list to string
            str_gpus = ','.join([str(g[0]) for g in gpus])
            # set environment variable
            os.environ['CUDA_VISIBLE_DEVICES'] = str_gpus
            phy_gpus = tf.config.experimental.list_physical_devices('GPU')
            for pg in phy_gpus:
                tf.config.experimental.set_memory_growth(pg, True)
            tf.keras.backend.set_image_data_format('channels_first')
            if self.param['rank']<=0:
                print('+' + 20 * '-' + '+')
                print('| Reside in Devices: |')
                print('+' + 70 * '-' + '+')
                for g in gpus:
                    print('| \033[1;36mGPU {:<2d}\033[0m | {:<25s} | {:>5d} MiB free out of {:<5d} MiB |'.format(
                        g[0], g[1], g[3], g[2]
                    ))
                    print('+' + 70 * '-' + '+')
        elif self.param['device'].lower() == 'cpu':
            if self.param['rank'] <= 0:
                print('+' + (23 * '-') + '+')
                print('| Reside in Device: \033[1;36mCPU\033[0m |')
                print('+' + (23 * '-') + '+')
        else:
            raise KeyError('NEBULAE ERROR ⨷ given device should be either cpu or gpu.')

        from ..astrobase import dock
        dock.coat = self.coat
        dock.shell = self.shell

    def coat(self, datum, as_const=True):
        if not (tf.is_tensor(datum) or isinstance(datum, PerReplica)): # numpy array
            datum = tf.convert_to_tensor(datum)
        if isinstance(datum, PerReplica):
            datum = tf.concat(datum.values, axis=0)

        if not as_const:
            datum = tf.Variable(datum)

        return datum

    def shell(self, datum, as_np=True):
        if isinstance(datum, PerReplica):
            datum = tf.concat(datum.values, axis=0)
        if as_np:
            datum = datum.numpy()
        return datum