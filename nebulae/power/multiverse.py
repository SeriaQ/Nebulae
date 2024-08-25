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
import sys
import torch.multiprocessing as mp
from ..kit.utility import ver2num
from ..rule import ENV_RANK

SG = 20
DP = 21
DT = 22


if ver2num(torch.__version__) >= ver2num('1.6.0'):
    is_new_version = True
else:
    is_new_version = False

if is_new_version:
    DDP = torch.nn.parallel.DistributedDataParallel
else:
    try:
        from apex import parallel
    except ImportError:
        from torch.nn import parallel
        print('NEBULAE WARNING ◘ The PyTorch version is lower than 1.6 which may cause abnormal BNs in distributed manner.')
    DDP = parallel.DistributedDataParallel

if ver2num(torch.__version__) >= ver2num('1.9.0'):
    from torch.distributed.run import parse_args
    from torch.distributed.run import run as ptrun
else:
    print('NEBULAE WARNING ◘ The PyTorch version is lower than 1.9, therefore the Multiverse is not recommended for use.')



class Multiverse(object):
    '''
    Args:
    universe: main function for a subprocess
    nworld: world size
    '''

    def __init__(self, universe, ncraft=1, nworld=1, mode=SG):
        self.universe = universe
        self.nworld = nworld
        self.ncraft = ncraft
        self.mode = mode
        self.rank = int(os.environ.get(ENV_RANK, -1))
        self.ddp_args = parse_args(sys.argv)
        self.ddp_args.master_port = 12345
        self.ddp_args.nnodes = str(nworld)
        self.ddp_args.nproc_per_node = str(ncraft)

    def __call__(self, *args, **kwargs):
        if self.nworld * self.ncraft == 1 or self.mode == SG or self.mode == DP:
            return self.universe(*args, **kwargs)
        
        if self.rank < 0:
            ptrun(self.ddp_args)
        else:
            return self.universe(*args, **kwargs)



class Universe(object):
    def __init__(self, mode=SG):
        self.rank = int(os.environ.get(ENV_RANK, -1))
        self.nworld = int(os.environ.get('WORLD_SIZE', -1))
        self.mode = mode
        self._mode_validity_check()
        if self.mode == DT:
            torch.distributed.init_process_group(backend="nccl", rank=self.rank, world_size=self.nworld)

    def __del__(self):
        if self.rank == 0:
            torch.distributed.destroy_process_group()

    def _mode_validity_check(self):
        if self.mode == SG:
            assert self.rank < 0
        elif self.mode == DP:
            assert self.rank < 0
        elif self.mode == DT:
            assert self.rank >= 0
        else:
            raise ValueError('NEBULAE ERROR ៙ current mode assigned to Universe is wrong: %d'%self.mode)

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

        if self.mode == DP:
            _model = torch.nn.DataParallel(_model)
        elif self.mode == DT:
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
        if self.mode == SG:
            return models
        
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
        if self.rank < 0:
            return tensors
        
        if not isinstance(tensors, (list, tuple)):
            return self._reduce(tensors)

        reduced_ts = []
        for t in tensors:
            reduced_ts.append(self._reduce(t))
        return tuple(reduced_ts)





# class Multiverse(object):
#     '''
#     Args:
#     universe: main function for a subprocess
#     nworld: world size
#     '''

#     def __init__(self, universe, nworld=1):
#         self.universe = universe
#         self.nworld = nworld
#         self.rank = -1
#         self.env = os.environ.copy()
#         self.env["MASTER_ADDR"] = '127.0.0.1'
#         self.env["MASTER_PORT"] = '12345'
#         self.env["WORLD_SIZE"] = str(nworld)
#         # self.env["OMP_NUM_THREADS"] = '1'

#     def __del__(self):
#         if self.rank == 0:
#             torch.distributed.destroy_process_group()

#     def __call__(self, *args, **kwargs):
#         # mp.set_start_method('spawn')
#         ps = []
#         def servo(*args, **kwargs):
#             def _get_univ():
#                 self.init()
#                 return self
#             from .. import power
#             power.Universe = _get_univ
#             return self.universe(*args, **kwargs)
#         for r in range(self.nworld):
#             self.env[ENV_RANK] = str(r)
#             self.env['LOCAL_RANK'] = str(r)
#             # p = mp.Process(target=self.universe, args=args, kwargs=kwargs)
#             p = mp.Process(target=servo, args=args, kwargs=kwargs)
#             p.start()
#             ps.append(p)
#         for p in ps:
#             p.join()

#         # mp.spawn(self.universe, args=(self,)+args, nprocs=self.nworld)

#     def init(self):
#         for k, v in self.env.items():
#             os.environ[k] = v
#         self.rank = int(os.environ[ENV_RANK])
#         torch.distributed.init_process_group(backend="nccl", rank=self.rank, world_size=self.nworld)

#     def _sync(self, model):
#         from ..astro.craft import Craft, EMA
#         if isinstance(model, EMA):
#             _model = model.hull
#         else:
#             _model = model

#         if isinstance(_model, Craft):
#             scope = _model.scope
#         else:
#             scope = ''

#         if is_new_version:
#             _model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(_model)
#             _model = DDP(_model, device_ids=[self.rank], output_device=self.rank)
#         else:
#             _model = parallel.convert_syncbn_model(_model)
#             _model = DDP(_model, delay_allreduce=True)

#         if scope:
#             _model.scope = scope

#         if isinstance(model, EMA):
#             model.hull = _model
#         else:
#             model = _model
#         return model

#     def sync(self, models):
#         if not isinstance(models, (list, tuple)):
#             return self._sync(models)

#         synced_md = []
#         for m in models:
#             synced_md.append(self._sync(m))

#         return tuple(synced_md)

#     def _reduce(self, tensor, aggregate=False):
#         rt = tensor.clone()
#         torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
#         if not aggregate:
#             rt /= self.nworld
#         return rt

#     def reduce(self, tensors, aggregate=False):
#         if not isinstance(tensors, (list, tuple)):
#             return self._reduce(tensors)

#         reduced_ts = []
#         for t in tensors:
#             reduced_ts.append(self._reduce(t))
#         return tuple(reduced_ts)



# class Universe(object):
#     def __init__(self):
#         pass

#     def init(self):
#         pass

#     def sync(self, models):
#         return models

#     def reduce(self, tensors, aggregate=False):
#         return tensors