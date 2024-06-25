#!/usr/bin/env python
'''
Created by Seria at 02/11/2018 3:38 PM
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
import csv
import h5py
from collections import abc
from multiprocessing import cpu_count
from math import ceil
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torch import cuda

from ..power import GPU
from ..rule import ENV_RANK, CHAR_SEP, FIELD_SEP
from ..kit.utility import ver2num

__all__ = ('Tank', 'Depot', 'load_h5', 'load_csv')


if ver2num(torch.__version__) >= ver2num('1.7.0'):
    is_new_version = True
else:
    is_new_version = False


def load_h5(data_path):
    assert data_path.endswith('h5') or data_path.endswith('hdf5')
    hdf5 = h5py.File(data_path, 'r')
    return hdf5

def load_csv(data_path, data_spec):
    assert data_path.endswith('csv') or data_path.endswith('txt')
    tdata = {}
    with open(data_path, 'r') as csvf:
        csvr = csv.reader(csvf, delimiter=CHAR_SEP, quotechar=FIELD_SEP)
        for l, line in enumerate(csvr):
            if l == 0:
                keys = line
                tdata = {k: [] for k in keys}
                continue
            for i, val in enumerate(line):
                if data_spec[keys[i]].startswith('int'):
                    tdata[keys[i]].append(int(val))
                elif data_spec[keys[i]].startswith('float'):
                    tdata[keys[i]].append(float(val))
                else:
                    tdata[keys[i]].append(val)
    return tdata



class Tank(Dataset):
    def __init__(self, *args, **kwargs):
        self._length = self.load(*args, **kwargs)

    def load(self, *args, **kwargs) -> int:
        raise NotImplementedError

    def fetch(self, idx):
        raise NotImplementedError

    def collate(self, batch):
        return default_collate(batch)

    def __getitem__(self, item):
        return self.fetch(item)

    def __len__(self):
        return self._length


class Depot(object):
    def __init__(self, engine=None):
        self.rank = int(os.environ.get(ENV_RANK, -1))
        self.nworld = int(os.environ.get('WORLD_SIZE', 1))
        if engine is not None and hasattr(engine, 'chip'):
            if engine.device == GPU and self.rank < 0: # only Data Parallel
                self._chip = torch.device('cuda')
            else:
                self._chip = engine.chip[self.rank]
            self._coater = engine.coat
        elif isinstance(engine, torch.device) and str(engine) != 'cpu':
            self._chip = engine
            def coat(t):
                t.to(self._chip)
                return t
            self._coater = coat
        else:
            self._chip = None
        self._tanks = {}
        self._batch_size = {}
        self._sampler = {} # only store tanks with shuffling
        self.MPE = {}

    def mount(self, tank, batch_size, shuffle=True, in_same_size=True, nworker=-1, prefetch=0, transmit=False, collater=None):
        # >| get unique id
        tid = '0t' + hex(id(tank))[2:]
        # >| check argument validity
        if tid in self._tanks.keys():
            raise AttributeError('NEBULAE ERROR ៙ this tank has already been mounted.')
        if nworker == 0 and prefetch > 0:
            raise ValueError('NEBULAE ERROR ៙ the number of workers should be more than 1 when prefetcher is enabled.')
        assert nworker != 0, 'NEBULAE ERROR ៙ the number of workers must be a positive integer.'
        if transmit and (self._chip is None or not cuda.is_available()):
            print('NEBULAE WARNING ◘ you have no GPU to which the data is transmitting.')
            transmit = False
        self._transmit = transmit
        if nworker == 1:
            prefetch = -1
        else:
            if is_new_version:
                prefetch = max(1, prefetch)  # at least be 1 in pytorch
            else:
                print('NEBULAE WARNING ◘ to prefetch on CPU side, please upgrade PyTorch to 1.7.0 or higher.')
                prefetch = -1

        # >| mount dataset
        self._batch_size[tid] = batch_size
        if in_same_size:
            self.MPE[tid] = len(tank) // (batch_size * self.nworld)
        else:
            self.MPE[tid] = ceil(len(tank) / (batch_size * self.nworld))
        nworker = cpu_count() if nworker <= 0 else nworker - 1
        if hasattr(tank, 'collate'):
            if collater is not None:
                print('NEBULAE WARNING ◘ tank.collate has the privilege of overwriting input collater.')
            collater = tank.collate
        if self.rank >= 0:
            from torch.utils.data import distributed as dist
            sampler = dist.DistributedSampler(tank)
            if prefetch < 0:
                loader = DataLoader(tank, batch_size, sampler=sampler, collate_fn=collater,
                                          drop_last=in_same_size, num_workers=nworker)
            else:
                loader = DataLoader(tank, batch_size, sampler=sampler, collate_fn=collater,
                                    drop_last=in_same_size, num_workers=nworker, prefetch_factor=prefetch)
            if shuffle:
                sampler.set_epoch(0)
                self._sampler[tid] = sampler
        else:
            if prefetch < 0:
                loader = DataLoader(tank, batch_size, shuffle, collate_fn=collater,
                                          drop_last=in_same_size, num_workers=nworker)
            else:
                loader = DataLoader(tank, batch_size, shuffle, collate_fn=collater,
                                    drop_last=in_same_size, num_workers=nworker, prefetch_factor=prefetch)
        # >| prefetch if necessary
        if self._transmit: # no matter how many batches to prefetch on GPU is all the same
            self._tanks[tid] = [tank, loader, loader.__iter__(), 0, 0, cuda.Stream(device=self._chip)]
            self._prefetch(tid)
        else:
            self._tanks[tid] = [tank, loader, loader.__iter__(), 0, 0]
        return tid

    def unmount(self, tid):
        assert tid in self._tanks.keys(), 'NEBULAE ERROR ៙ this tank is not mounted in the depot.'
        self._tanks.pop(tid)
        self._batch_size.pop(tid)
        self.MPE.pop(tid)
        if tid in self._sampler.keys():
            self._sampler.pop(tid)

    def _prefetch(self, tid):
        self._fetched_data = self._fetch(tid)
        with cuda.stream(self._tanks[tid][-1]):
            # self._fetched_data = self._fetched_data.cuda(device=self.chip, non_blocking=True)
            if isinstance(self._fetched_data, torch.Tensor):
                self._fetched_data = self._coater(self._fetched_data, sync=False)
            elif isinstance(self._fetched_data, abc.Mapping):
                self._fetched_data = {key: self._coater(self._fetched_data[key]) for key in self._fetched_data}
            elif isinstance(self._fetched_data, abc.Sequence):
                self._fetched_data = [self._coater(fd) for fd in self._fetched_data]
            else:
                raise TypeError('NEBULAE ERROR ៙ batchified data should be a tensor, sequence or dictionary but is %s'
                                % type(self._fetched_data))

    def _fetch(self, tid):
        loader, iterator, mile, epoch = self._tanks[tid][1:5]
        if mile == self.MPE[tid]:
            iterator = loader.__iter__()
            self._tanks[tid][2:4] = iterator, 1
            self._tanks[tid][4] += 1
            if tid in self._sampler.keys():
                self._sampler[tid].set_epoch(epoch + 1)
        else:
            self._tanks[tid][3] += 1
        return iterator.__next__()

    def _next(self, tid):
        if self._transmit:
            cuda.current_stream().wait_stream(self._tanks[tid][-1])
            fetched_data = self._fetched_data
            self._prefetch(tid)
            return fetched_data
        else:
            return self._fetch(tid)

    def next(self, tid=None):
        if tid is None:
            ret = {}
            for k in self._tanks.keys():
                ret[k] = self._next(k)
            return ret
        else:
            return self._next(tid)