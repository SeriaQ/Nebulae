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
import io
import random as rand
import numpy as np
import h5py
from PIL import Image

from ..toolkit.decorator import Timer

class FuelTank(object):
    def __init__(self, param):
        self.param = param
        self.epoch = 0
        duration = self._loadData()
        print('+' + (61 * '-') + '+')
        print('| \033[1;35m%-17s\033[0m fuel tank has been mounted within \033[1;35m%6.3fs\033[0m |'
              % (param['name'], duration))
        print('+' + (61 * '-') + '+')

    def __del__(self):
        self.datafile.close()
        print('+' + (50 * '-') + '+')
        print('| \033[1;35m%-17s\033[0m fuel tank is no longer mounted |' % self.param['name'])
        print('+' + (50 * '-') + '+')

    def _shuffleData(self):
        '''
        Args:
        num: batch size
        leftover: the rest samples of last epoch
        '''
        nsample = self.nsample
        for j in range(nsample):
            idx = int(rand.random() * (nsample - j)) + j
            temp = self.order[j]
            self.order[j] = self.order[idx]
            self.order[idx] = temp

    def _dataAugmentation(self, source):
        def _spatialAugmentation(img, da_type, theta):
            data_aug_err = Exception('%s spatial augmentation is unsupported.' % da_type)
            if da_type == 'flip':
                aug_img = np.zeros(img.shape)
                if self.param['channel'] == 1:
                    aug_img = np.fliplr(img)
                else:
                    for c in range(self.param['channel']):
                        aug_img[:, :, c] = np.fliplr(img[:, :, c])
                return aug_img
            elif da_type == 'brightness':
                aug_img = theta + img  # pixels range between [theta, 1+theta]
                aug_img = np.clip(aug_img, 0, 1)  # pixels range between [0, 1]
                return aug_img
            elif da_type == 'gamma_contrast':
                aug_img = theta * (img - 0.5) # pixels range between [0, theta]
                aug_img = np.clip(aug_img, -0.5, 0.5) + 0.5 # pixels range between [0, 1]
                return aug_img
            elif da_type == 'log_contrast':
                aug_img = theta * np.log(img + 1) # pixels range between [0, theta * ln2]
                aug_img = np.where(aug_img > 1, 1, aug_img)  # pixels range between [0, 1]
                return aug_img
            else:
                raise data_aug_err

        def _temporalAugmentation(seq, da_type, theta):
            data_aug_err = Exception('%s temporal augmentation is unsupported.' % da_type)
            if da_type == 'sample':
                seq_len = self.nsample // theta # here theta is sampling frequency
                aug_seq = np.zeros((seq_len,) + seq.shape[1:])
                for i in range(seq_len):
                    aug_seq[i] = seq[int((i + rand.random()) * theta)]
                return aug_seq
            else:
                raise data_aug_err

        if self.param['is_seq']: # if this call is for sequential data augmentation
            for t,ta in enumerate(self.param['temporal_aug'].split(',')):
                if not ta:
                    continue
                if rand.random() < self.param['p_ta'][t]:
                    source = _temporalAugmentation(source, ta, self.param['theta_ta'][t])
                    for s, sa in enumerate(self.param['spatial_aug'].split(',')):
                        if not sa:
                            continue
                        if rand.random() < self.param['p_sa'][s]:
                            for i in range(source.shape[0]):
                                source[i] = _spatialAugmentation(source[i], sa, self.param['theta_sa'][t])
        else:
            for s,sa in enumerate(self.param['spatial_aug'].split(',')):
                if not sa:
                    continue
                if rand.random() < self.param['p_sa'][s]:
                    source = _spatialAugmentation(source, sa, self.param['theta_sa'][s])

        return source * 2 - 1 # pixels range between [-1, 1]

    @Timer
    def _loadData(self):
        # --------------------------------- load data --------------------------------- #
        format = self.param['data_path'].split('.')[-1]
        if format == 'npz':
            self.datafile = np.load(self.param['data_path'])
            self.attributes = self.datafile.keys()
            self.dtypes = {}
            for atr in self.attributes:
                if atr == self.param['data_key'] and self.param['is_compressed']:
                    self.dtypes[atr] = np.dtype('float32')
                else:
                    self.dtypes[atr] = self.datafile[atr].dtype

        elif format == 'hdf5':
            self.datafile = h5py.File(self.param['data_path'], 'r')
            self.attributes = self.datafile.keys()
            self.dtypes = {}
            for atr in self.attributes:
                if atr == self.param['data_key'] and self.param['is_compressed']:
                    self.dtypes[atr] = np.dtype('float32')
                else:
                    self.dtypes[atr] = self.datafile[atr].dtype

        # get number of samples and channels
        self.nsample = self.datafile[self.param['data_key']].shape[0]

        # --------------------------- initialize and shuffle --------------------------- #
        order = [i for i in range(self.nsample )]
        self.curr_idx = 0
        self.order = order
        if self.param['if_shuffle']:
            self._shuffleData()

    def _byte2Array(self, data_src):
        if not self.param['is_compressed']: # no need to recover data from bytes
            return data_src
        data_bytes = data_src.tobytes()
        data_pil = Image.open(io.BytesIO(data_bytes))
        if self.param['width']>0 and self.param['height']>0:
            # if need lower the image resolution
            if self.param['resol_ratio'] < 1:
                data_pil = data_pil.resize((int(self.param['resol_ratio'] * self.param['width']),
                                            int(self.param['resol_ratio'] * self.param['height'])))
            data_pil = data_pil.resize((self.param['width'], self.param['height']))
        data_np = np.array(data_pil) # pixels range between [0, 255]
        data_np = (data_np/255).astype(np.float32) # pixels range between [0, 1]
        if self.param['channel'] == 1:
            data_np = np.expand_dims(data_np, -1)
        return data_np

    def _fetchBatches(self):
        '''
        Fetch next batch
        '''
        while 1:
            self.curr_idx += self.param['batch_size']
            curr_batch = {}
            for atr in self.attributes:
                curr_batch[atr] = []
            # start a new epoch
            low_power = False
            # import time
            # start = time.time()
            if self.curr_idx > self.nsample:
                low_power = True
                self.epoch = -self.epoch
                # if the samples left are not enough to feed a mini batch, let's say n samples.
                # the current batch should be the last n samples in addition to the first (bs-n) samples
                for idx in self.order[self.curr_idx - self.param['batch_size']:self.nsample]:
                    for atr in self.attributes:
                        if atr == self.param['data_key']:
                            curr_batch[atr] += [self._dataAugmentation(self._byte2Array(self.datafile[atr][idx]))]
                        else:
                            curr_batch[atr] += [self.datafile[atr][idx]]
                for idx in self.order[0:self.curr_idx - self.nsample]:
                    for atr in self.attributes:
                        if atr == self.param['data_key']:
                            curr_batch[atr] += [self._dataAugmentation(self._byte2Array(self.datafile[atr][idx]))]
                        else:
                            curr_batch[atr] += [self.datafile[atr][idx]]

                if self.param['if_shuffle']:
                    # reshuffle data to read
                    self._shuffleData()
                else:
                    self.order = [i for i in range(self.nsample)]
                self.curr_idx = 0
            else:
                for idx in self.order[self.curr_idx - self.param['batch_size']:self.curr_idx]:
                    for atr in self.attributes:
                        if atr == self.param['data_key']:
                            curr_batch[atr] += [self._dataAugmentation(self._byte2Array(self.datafile[atr][idx]))]
                        else:
                            curr_batch[atr] += [self.datafile[atr][idx]]

            for atr in self.attributes:
                curr_batch[atr] = np.array(curr_batch[atr]).astype(self.dtypes[atr])
            # print('%.3fs'%(time.time()-start))
            yield curr_batch
            if low_power:
                # epoch counter turns positive when an epoch ends
                self.epoch = abs(self.epoch) + 1