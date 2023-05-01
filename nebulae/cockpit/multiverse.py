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
import torch
import os
import multiprocessing as mp
from ..kit.utility import ver2num

if ver2num(torch.__version__) >= ver2num('1.6.0'):
    is_new_version = True
else:
    is_new_version = False

if is_new_version:
    class DDP(torch.nn.parallel.DistributedDataParallel):
        def __init__(self, module, device_ids, output_device):
            super(DDP, self).__init__(module, device_ids=device_ids, output_device=output_device,
                                      find_unused_parameters=True)

        def __getattr__(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    return _parameters[name]
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name]
            if '_modules' in self.__dict__:
                modules = self.__dict__['_modules']
                if name in modules:
                    return modules[name]
            if hasattr(self, 'module'):
                if hasattr(self.module, name):
                    return getattr(self.module, name)
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))
else:
    try:
        from apex import parallel
    except ImportError:
        from torch.nn import parallel
        print('NEBULAE WARNING ◘ The PyTorch version is lower than 1.6 which may cause abnormal BNs in distributed manner.')
    class DDP(parallel.DistributedDataParallel):
        def __init__(self, module, delay_allreduce):
            super(DDP, self).__init__(module, delay_allreduce=delay_allreduce)

        def __getattr__(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    return _parameters[name]
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name]
            if '_modules' in self.__dict__:
                modules = self.__dict__['_modules']
                if name in modules:
                    return modules[name]
            if hasattr(self, 'module'):
                if hasattr(self.module, name):
                    return getattr(self.module, name)
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))


# DDP = parallel.DistributedDataParallel

class Multiverse(object):
    '''
    Args:
    nworld: world size
    '''

    def __init__(self, universe, nworld=1):
        self.universe = universe
        self.nworld = nworld
        self.rank = -1
        self.env = os.environ.copy()
        self.env["MASTER_ADDR"] = '127.0.0.1'
        self.env["MASTER_PORT"] = '29500'
        self.env["WORLD_SIZE"] = str(nworld)
        self.env["OMP_NUM_THREADS"] = '1'

    def __call__(self, *args, **kwargs):
        # mp.set_start_method('spawn')
        ps = []
        for r in range(self.nworld):
            self.rank = r
            self.env['RANK'] = str(r)
            self.env['LOCAL_RANK'] = str(r)
            p = mp.Process(target=self.universe, args=(self,)+args, kwargs=kwargs)
            p.start()
            ps.append(p)
        for p in ps:
            p.join()

    def init(self):
        for k, v in self.env.items():
            os.environ[k] = v

    def _sync(self, model):
        scope = model.scope
        if is_new_version:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(torch.device('cuda:%d' % self.rank))
            model = DDP(model, device_ids=[self.rank], output_device=self.rank)
        else:
            model = parallel.convert_syncbn_model(model)
            model = model.to(torch.device('cuda:%d' % self.rank))
            model = DDP(model, delay_allreduce=True)

        model.scope = scope
        return model

    def sync(self, models):
        if not isinstance(models, (list, tuple)):
            return self._sync(models)

        synced_md = []
        for m in models:
            synced_md.append(self._sync(m))

        return tuple(synced_md)

    def reduce(self, tensor, aggregate=False):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        if not aggregate:
            rt /= self.nworld
        return rt