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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import h5py
from collections import abc
from math import ceil
import tensorflow as tf
# import tensorflow_io as tfio


class Tank(object):
    def __init__(self, data_path, data_specf, batch_size, shuffle=True, in_same_size=True,
                 fetch_fn=None, prep_fn=None, collate_fn=None):
        name = os.path.basename(data_path).split('.')[0]
        self.rank = int(os.environ.get('RANK', -1))
        # nworld = int(os.environ.get('WORLD_SIZE', 1))

        ret_struct = -1
        with h5py.File(data_path, 'r') as f:
            length = len(f[list(data_specf.keys())[0]])
            elem = fetch_fn(f, 0)
            etype = type(elem)
            # ndarray
            if etype.__module__ == 'numpy' and etype.__name__ != 'str_' and etype.__name__ != 'string_':
                ret_struct = 0
                assert len(data_specf) == 1
                for k, v in data_specf.items():
                    if v.startswith('v'):
                        v = 'string'
                    dtype = getattr(tf.dtypes, v)
                    # dshape = elem.shape
            # single value
            elif isinstance(elem, (int, float, str, bytes)):
                ret_struct = 1
                assert len(data_specf) == 1
                for k, v in data_specf.items():
                    if v.startswith('v'):
                        v = 'string'
                    dtype = getattr(tf.dtypes, v)
                    # dshape = ()
            # mapping type
            elif isinstance(elem, abc.Mapping):
                ret_struct = 2
                dtype = {}
                dshape = {}
                for k, v in data_specf.items():
                    if v.startswith('v'):
                        v = 'string'
                    dtype[k] = getattr(tf.dtypes, v)
                    # dshape[k] = elem[k].shape
            # sequential type
            elif isinstance(elem, abc.Sequence):
                ret_struct = 3
                dtype = []
                dshape = []
                for i, v in enumerate(data_specf.values()):
                    if v.startswith('v'):
                        v = 'string'
                    dtype.append(getattr(tf.dtypes, v))
                    # dshape.append(elem[i].shape)
            else:
                raise Exception('NEBULAE ERROR ⨷ %s is not a valid type of data.' % type(elem))

        class TData(tf.data.Dataset):
            @staticmethod
            def _generator():
                # Opening the file
                hdf5 = h5py.File(data_path, 'r')
                length = len(hdf5[list(data_specf.keys())[0]])

                for idx in range(length):
                    # Reading data from the file
                    ret = fetch_fn(hdf5, idx)
                    for i, (k, v) in enumerate(data_specf.items()):
                        if v.startswith('v'):
                            if ret_struct <= 1:
                                ret = ret.tobytes()
                            elif ret_struct == 2:
                                ret[k] = ret[k].tobytes()
                            else:
                                ret[i] = ret[i].tobytes()
                    yield ret

            def __new__(cls):
                dset = tf.data.Dataset.from_generator(cls._generator, output_types=dtype)#, output_shapes=dshape)
                    # output_signature=[tf.TensorSpec(shape=(), dtype=tf.int32)])
                dset.length = length
                if self.rank<=0:
                    print('+' + (49 * '-') + '+')
                    print('| \033[1;35m%-20s\033[0m fuel tank has been mounted |' % name)
                    print('+' + (49 * '-') + '+')
                return dset

        self.name = name
        self.counter = 0

        self.tdata = TData()
        length = self.tdata.length
        if prep_fn is not None:
            self.tdata = self.tdata.map(prep_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if shuffle:
            self.tdata = self.tdata.shuffle(8 * batch_size, reshuffle_each_iteration=True)
        self.tdata = self.tdata.batch(batch_size, drop_remainder=in_same_size)
        self.batch_size = batch_size
        if in_same_size:
            self.MPE = length // batch_size
        else:
            self.MPE = ceil(length / batch_size)
        self.tdata = self.tdata.cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

    def __del__(self):
        if self.rank <= 0:
            print('+' + (53 * '-') + '+')
            print('| \033[1;35m%-20s\033[0m fuel tank is no longer mounted |' % self.name)
            print('+' + (53 * '-') + '+')

    def __len__(self):
        return self.tdata.length

    def next(self):
        if self.counter == 0: # create a new iterator at the begining of an epoch
            self.iterator = self.tdata.__iter__()
        self.counter += 1
        if self.counter == self.MPE:
            self.counter = 0
        return self.iterator.__next__()