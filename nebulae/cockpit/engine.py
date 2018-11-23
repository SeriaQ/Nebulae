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

import tensorflow as tf
import subprocess as subp

class Engine(object):
    '''
    Param:
    device: 'gpu' or 'cpu'
    available_gpus
    gpu_mem_fraction
    least_mem
    '''
    def __init__(self, param):
        self.param = param
        # look for available gpu devices
        config_proto = tf.ConfigProto(log_device_placement=False)
        if self.param['device'].lower() == 'gpu':
            config_proto.gpu_options.per_process_gpu_memory_fraction = self.param['gpu_mem_fraction']
            if not self.param['available_gpus']:
                gpus = self.getAvailabelGPU()
                if gpus < 0:
                    raise Exception('No available gpu', gpus)
                else:
                    self.param['available_gpus'] = str(gpus)
            config_proto.gpu_options.visible_device_list = self.param['available_gpus']
            print('+' + (25 * '-') + '+')
            print('| Reside in Device: \033[1;36mGPU-%s\033[0m |' % self.param['available_gpus'])
            print('+' + (25 * '-') + '+')
        elif self.param['device'].lower() == 'cpu':
            print('+' + (23 * '-') + '+')
            print('| Reside in Device: \033[1;36mCPU\033[0m |')
            print('+' + (23 * '-') + '+')
        else:
            raise KeyError('Given device should be either cpu or gpu.')
        # start a session
        self.sess = tf.Session(config=config_proto)

    def getAvailabelGPU(self):
        p = subp.Popen('nvidia-smi', stdout=subp.PIPE)
        gpu_id = 0  # next gpu we are about to probe
        flag_gpu = False
        max_occupied = self.param['least_mem']
        id_best = -1  # gpu having max avialable memory
        for l in p.stdout.readlines():
            line = l.decode('utf-8').split()
            if len(line) < 1:
                break
            elif len(line) < 2:
                continue
            if line[1] == str(gpu_id):
                flag_gpu = True
                continue
            if flag_gpu:
                occupancy = int(line[8].split('M')[0])
                if occupancy < max_occupied:
                    max_occupied = occupancy
                    id_best = gpu_id
                gpu_id += 1
                flag_gpu = False
        return id_best