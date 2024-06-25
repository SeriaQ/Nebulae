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
from ..kit.utility import GPUtil, ver2num
from ..rule import ENV_RANK

CPU = 0
GPU = 1



class Engine(object):
    '''
    Param:
    device: CPU or GPU
    available_gpus
    gpu_mem_fraction
    if_conserve
    least_mem
    '''
    def __init__(self, device=GPU, ngpu=1, least_mem=1024, avail_gpus=()):
        self.rank = int(os.environ.get(ENV_RANK, -1))
        self.device = device
        # look for available gpu devices
        if self.device == GPU:
            if len(avail_gpus) == 0:
                gputil = GPUtil()
                gpus = gputil.available(ngpu, least_mem)
            else:
                gpus = avail_gpus
            if len(gpus) < ngpu:
                raise Exception('NEBULAE ERROR ៙ no enough available gpu.')
            # convert gpu list to string
            str_gpus = ','.join([str(g[0]) for g in gpus])
            # set environment variable
            os.environ['CUDA_VISIBLE_DEVICES'] = str_gpus
            self.chip = [torch.device('cuda:%d'%i) for i in range(ngpu)]
            # setup multi-gpu environment
            if self.rank>=0:
                torch.backends.cudnn.benchmark = True
                torch.cuda.set_device(self.rank)
            if self.rank<=0:
                print('+' + 20 * '-' + '+')
                print('| Reside in Devices: |')
                print('+' + 70 * '-' + '+')
                for g in gpus:
                    print('| \033[1;36mGPU {:<2d}\033[0m | {:<25s} | {:>5d} MiB free out of {:<5d} MiB |'.format(
                        g[0], g[1], g[3], g[2]
                    ))
                    print('+' + 70 * '-' + '+')
        elif self.device == CPU:
            assert self.rank<0
            self.chip = [torch.device('cpu')]
            print('+' + (24 * '-') + '+')
            print('| Reside in Devices: \033[1;36mCPU\033[0m |')
            print('+' + (24 * '-') + '+')
        else:
            raise KeyError('NEBULAE ERROR ៙ given device should be either cpu or gpu.')

        ###################################
        ##          IT WORKS !           ##
        ###################################
        from ..astro import dock
        dock.coat = self.coat
        dock.shell = self.shell

    def coat(self, datum, as_const=True, sync=True):
        if not isinstance(datum, torch.Tensor): # numpy array
            if isinstance(datum, (int, float)) or datum.shape==(): # scalar
                datum = torch.tensor(datum)
            else:
                datum = torch.from_numpy(datum)

        if as_const:
            datum.requires_grad = False
        else:
            datum.requires_grad = True

        if self.device == GPU:
            if self.rank<0:
                return datum.cuda(non_blocking=not sync)
            else:
                return datum.to(self.chip[self.rank], non_blocking=not sync)
        elif self.device == CPU:
            return datum.cpu()
        else:
            raise KeyError('NEBULAE ERROR ៙ given device should be either cpu or gpu.')

    def shell(self, datum, as_np=True, sync=False):
        datum = datum.detach()
        # add a sync point
        if sync and self.device == GPU:
            if self.rank < 0:
                torch.cuda.synchronize()
            else:
                torch.cuda.synchronize(self.chip[self.rank])
        if as_np:
            if datum.size == 1:
                datum = datum.cpu().numpy()[0]
            else:
                datum = datum.cpu().numpy()
        return datum