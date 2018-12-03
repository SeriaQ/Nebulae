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

from math import ceil

from .fuel_tank import FuelTank

class FuelDepot(object):
    def __init__(self):
        self.dataset = {} # contains all fuel tanks
        self.fetcher = {} # contains batch fetcher of all datasets
        self.modifiable_keys = ['name', 'batch_size', 'height', 'width', 'channel',
                                'resol_ratio', 'spatial_aug', 'p_sa', 'temporal_aug', 'p_ta']

    def loadFuel(self, config=None, name=None, batch_size=None, if_shuffle=True,
                 data_path=None, data_key=None, is_compressed=True,
                 height=0, width=0, channel=3, resol_ratio=1, is_seq=False,
                 spatial_aug='', p_sa=(0), theta_sa=(0),
                 temporal_aug='', p_ta=(0), theta_ta=(0)):
        if config:
            config['if_shuffle'] = config.get('if_shuffle', if_shuffle)
            config['is_compressed'] = config.get('is_compressed', is_compressed)
            config['height'] = config.get('height', height)
            config['width'] = config.get('width', width)
            config['channel'] = config.get('channel', channel)
            config['resol_ratio'] = config.get('resol_ratio', resol_ratio)
            config['is_seq'] = config.get('is_seq', is_seq)
            config['spatial_aug'] = config.get('spatial_aug', spatial_aug)
            config['p_sa'] = config.get('p_sa', p_sa)
            config['theta_sa'] = config.get('theta_sa', theta_sa)
            config['temporal_aug'] = config.get('temporal_aug', temporal_aug)
            config['p_ta'] = config.get('p_ta', p_ta)
            config['theta_ta'] = config.get('theta_ta', theta_ta)
        else:
            config = {}
            config['name'] = name
            config['batch_size'] = batch_size
            config['data_path'] = data_path
            config['data_key'] = data_key
            config['is_compressed'] = is_compressed
            config['if_shuffle'] = if_shuffle
            config['height'] = height
            config['width'] = width
            config['channel'] = channel
            config['resol_ratio'] = resol_ratio
            config['is_seq'] = is_seq
            config['spatial_aug'] = spatial_aug
            config['p_sa'] = p_sa
            config['theta_sa'] = theta_sa
            config['temporal_aug'] = temporal_aug
            config['p_ta'] = p_ta
            config['theta_ta'] = theta_ta
        if config['name'] in self.dataset:
            raise Exception('%s is already mounted.' % name)
        self.dataset[config['name']] = FuelTank(config)
        self.fetcher[config['name']] = self.dataset[config['name']]._fetchBatches()

    def unloadFuel(self, name):
        name_err = Exception('%s is not an mounted fuel tank.' % name)
        if name not in self.dataset.keys():
            raise name_err
        else:
            self.fetcher.pop(name)
            self.dataset.pop(name)

    def nextBatch(self, name):
        return self.fetcher[name].__next__()

    def currentEpoch(self, name):
        return abs(self.dataset[name].epoch)+1

    def stepsPerEpoch(self, name):
        return ceil(self.dataset[name].nsample/self.dataset[name].param['batch_size'])

    def editProperty(self, dataname, config=None, **kwargs):
        flag_rename = False
        if config:
            kwargs = config
        for key in kwargs:
            if key not in self.modifiable_keys:
                print('%s is not a modifiable parameter or has not been defined.'%key)
            elif key == 'name':
                flag_rename = True
                self.dataset[dataname].param[key] = kwargs[key]
            else:
                self.dataset[dataname].param[key] = kwargs[key]
        if flag_rename:
            self.dataset[kwargs['name']] = self.dataset[dataname]
            self.dataset.pop(dataname)
            self.fetcher[kwargs['name']] = self.fetcher[dataname]
            self.fetcher.pop(dataname)