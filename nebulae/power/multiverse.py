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
# import multiprocessing as mp
import torch.multiprocessing as mp
from ..kit.utility import ver2num
from ..rule import ENV_RANK

if ver2num(torch.__version__) >= ver2num('1.6.0'):
    is_new_version = True
else:
    is_new_version = False

if is_new_version:
    DDP = torch.nn.parallel.DistributedDataParallel
    # class DDP(torch.nn.parallel.DistributedDataParallel):
    #     def __init__(self, module, device_ids, output_device, check_unused=False):
    #         super(DDP, self).__init__(module, device_ids=device_ids, output_device=output_device,
    #                                   find_unused_parameters=check_unused)

    #     def __getattr__(self, name: str):
    #         if '_parameters' in self.__dict__:
    #             _parameters = self.__dict__['_parameters']
    #             if name in _parameters:
    #                 return _parameters[name]
    #         if '_buffers' in self.__dict__:
    #             _buffers = self.__dict__['_buffers']
    #             if name in _buffers:
    #                 return _buffers[name]
    #         if '_modules' in self.__dict__:
    #             modules = self.__dict__['_modules']
    #             if name in modules:
    #                 return modules[name]
            
    #         raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    #     def on(self):
    #         self.module.on()

    #     def off(self):
    #         self.module.off()

    #     def update(self):
    #         self.module.update()

    #     def vars(self):
    #         return self.module.vars()

    #     def weights(self):
    #         return self.module.weighs()
else:
    try:
        from apex import parallel
    except ImportError:
        from torch.nn import parallel
        print('NEBULAE WARNING ◘ The PyTorch version is lower than 1.6 which may cause abnormal BNs in distributed manner.')
    DDP = parallel.DistributedDataParallel
    # class DDP(parallel.DistributedDataParallel):
    #     def __init__(self, module, delay_allreduce):
    #         super(DDP, self).__init__(module, delay_allreduce=delay_allreduce)

    #     def __getattr__(self, name: str):
    #         if '_parameters' in self.__dict__:
    #             _parameters = self.__dict__['_parameters']
    #             if name in _parameters:
    #                 return _parameters[name]
    #         if '_buffers' in self.__dict__:
    #             _buffers = self.__dict__['_buffers']
    #             if name in _buffers:
    #                 return _buffers[name]
    #         if '_modules' in self.__dict__:
    #             modules = self.__dict__['_modules']
    #             if name in modules:
    #                 return modules[name]
    #         if hasattr(self, 'module'):
    #             if hasattr(self.module, name):
    #                 return getattr(self.module, name)
    #         raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))



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
        self.env["MASTER_PORT"] = '12345'
        self.env["WORLD_SIZE"] = str(nworld)
        # self.env["OMP_NUM_THREADS"] = '1'

    def __del__(self):
        if self.rank == 0:
            torch.distributed.destroy_process_group()

    def __call__(self, *args, **kwargs):
        # mp.set_start_method('spawn')
        ps = []
        def servo(*args, **kwargs):
            def _get_univ():
                self.init()
                return self
            from .. import power
            power.Universe = _get_univ
            return self.universe(*args, **kwargs)
        for r in range(self.nworld):
            self.env[ENV_RANK] = str(r)
            self.env['LOCAL_RANK'] = str(r)
            # p = mp.Process(target=self.universe, args=args, kwargs=kwargs)
            p = mp.Process(target=servo, args=args, kwargs=kwargs)
            p.start()
            ps.append(p)
        for p in ps:
            p.join()

        # mp.spawn(self.universe, args=(self,)+args, nprocs=self.nworld)

    def init(self):
        for k, v in self.env.items():
            os.environ[k] = v
        self.rank = int(os.environ[ENV_RANK])
        torch.distributed.init_process_group(backend="nccl", rank=self.rank, world_size=self.nworld)

    def _sync(self, model):
        from ..astro.craft import Craft, EMA
        if isinstance(model, EMA):
            _model = model.hull
        else:
            _model = model

        if isinstance(_model, Craft):
            scope = _model.scope
        else:
            scope = ''

        if is_new_version:
            _model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(_model)
            _model = DDP(_model, device_ids=[self.rank], output_device=self.rank)
        else:
            _model = parallel.convert_syncbn_model(_model)
            _model = DDP(_model, delay_allreduce=True)

        if scope:
            _model.scope = scope

        if isinstance(model, EMA):
            model.hull = _model
        else:
            model = _model
        return model

    def sync(self, models):
        if not isinstance(models, (list, tuple)):
            return self._sync(models)

        synced_md = []
        for m in models:
            synced_md.append(self._sync(m))

        return tuple(synced_md)

    def _reduce(self, tensor, aggregate=False):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        if not aggregate:
            rt /= self.nworld
        return rt

    def reduce(self, tensors, aggregate=False):
        if not isinstance(tensors, (list, tuple)):
            return self._reduce(tensors)

        reduced_ts = []
        for t in tensors:
            reduced_ts.append(self._reduce(t))
        return tuple(reduced_ts)



class Universe(object):
    def __init__(self):
        pass

    def init(self):
        pass

    def sync(self, models):
        return models

    def reduce(self, tensors, aggregate=False):
        return tensors