#!/usr/bin/env python
'''
engine_mx
Created by Seria at 12/02/2019 3:45 PM
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
import torch
from ..toolkit.utility import GPUtil

class EnginePT(object):
    '''
    Param:
    device: 'gpu' or 'cpu'
    available_gpus
    gpu_mem_fraction
    if_conserve
    least_mem
    '''
    def __init__(self, param):
        self.param = param
        # look for available gpu devices
        if self.param['device'].lower() == 'gpu':
            if len(self.param['avail_gpus']) == 0:
                gputil = GPUtil()
                gpus = gputil.available(self.param['ngpus'], self.param['least_mem'])
            else:
                gpus = self.param['avail_gpus']
            if len(gpus) < self.param['ngpus']:
                raise Exception('NEBULAE ERROR ⨷ no enough available gpu.')
            # convert gpu list to string
            str_gpus = ','.join([str(g[0]) for g in gpus])
            # set environment variable
            os.environ['CUDA_VISIBLE_DEVICES'] = str_gpus
            self.device = [torch.device('cuda:%d'%i) for i in range(self.param['ngpus'])]
            # setup multi-gpu environment
            if self.param['rank']>=0:
                torch.backends.cudnn.benchmark = True
                torch.cuda.set_device(self.param['rank'])
                torch.distributed.init_process_group(backend='nccl', init_method='env://')
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
            assert self.param['rank']<0
            self.device = ['cpu']
            print('+' + (24 * '-') + '+')
            print('| Reside in Devices: \033[1;36mCPU\033[0m |')
            print('+' + (24 * '-') + '+')
        else:
            raise KeyError('NEBULAE ERROR ⨷ given device should be either cpu or gpu.')

        ###################################
        ##          IT WORKS !           ##
        ###################################
        from ..astrobase import dock
        dock.coat = self.coat
        dock.shell = self.shell

    def coat(self, datum, as_const=True):
        if not isinstance(datum, torch.Tensor): # numpy array
            if datum.shape==(): # scalar
                datum = torch.tensor(datum)
            else:
                datum = torch.from_numpy(datum)

        if as_const:
            datum.requires_grad = False
        else:
            datum.requires_grad = True

        if self.param['device'] == 'gpu':
            if self.param['rank']<0:
                return datum.cuda()
            else:
                return datum.to(self.device[self.param['rank']])
        elif self.param['device'] == 'cpu':
            return datum.cpu()
        else:
            raise KeyError('NEBULAE ERROR ⨷ given device should be either cpu or gpu.')

    def shell(self, datum, as_np=True):
        datum = datum.detach()
        if as_np:
            if datum.size == 1:
                datum = datum.cpu().numpy()[0]
            else:
                datum = datum.cpu().numpy()
        return datum