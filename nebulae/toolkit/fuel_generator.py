#!/usr/bin/env python
'''
Created by Seria at 02/11/2018 3:00 PM
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

from PIL import Image
import io
import os
import csv
import h5py
import numpy as np

from .decorator import Timer

class FuelGenerator(object):

    def __init__(self, config=None, file_dir=None, file_list=None, dtype=None,
                 height=224, width=224, channel=1, encode='jpeg'):
        self.param = {}
        self.data = {}
        self.modifiable_keys = ['file_dir', 'file_list', 'dtype',
                                'height', 'width', 'channel', 'encode']
        self.valid_dtypes = ['uint8', 'uint16', 'uint32', 'int8', 'int16', 'int32', 'int64',
                             'float16', 'float32', 'float64', 'str', 'bool']
        if config:
            self.param['file_dir'] = config.get('file_dir')
            self.param['file_list'] = config.get('file_list')
            self.param['dtype'] = [np.dtype(dt) for dt in config.get('dtype')]
            self.param['height'] = config.get('height', height)
            self.param['width'] = config.get('width', width)
            self.param['channel'] = config.get('channel', channel)
            self.param['encode'] = config.get('encode', encode)
        else:
            self.param['file_dir'] = file_dir
            self.param['file_list'] = file_list
            self.param['dtype'] = dtype
            self.param['height'] = height
            self.param['width'] = width
            self.param['channel'] = channel
            self.param['encode'] = encode

        # check if key arguments are valid
        if self.param['file_list'].split('.')[-1] != 'csv':
            raise Exception('file list should be a csv file.')
        for dt in self.param['dtype']:
            if dt not in self.valid_dtypes:
                raise Exception('%s is not a valid data type.' % dt)


    def _preProcess(self, img_path):
        ch_err = Exception('Images having %d channels are invalid.' % self.param['channel'])
        if self.param['channel'] == 1:
            image = Image.open(img_path).convert('L')
        elif self.param['channel'] == 3:
            image = Image.open(img_path)
        else:
            raise ch_err
        image = image.resize((self.param['width'], self.param['height']))
        with io.BytesIO() as buffer:
            image.save(buffer, format=self.param['encode'])
            encoded_bytes = buffer.getvalue()
        np_bytes = np.frombuffer(encoded_bytes, dtype=np.uint8)

        return np_bytes

    @Timer
    def _file2Byte(self, dst_path):
        with open(os.path.join(self.param['file_dir'], self.param['file_list']), 'r') as filelist:
            content = csv.reader(filelist)
            for l, line in enumerate(content):
                if l == 0: # initialize data dict
                    self.info_keys = line
                    if len(line) != len(self.param['dtype']):
                        raise Exception('number of given dtypes does not match the provided csv file.')
                    for key in line:
                        self.data[key] = []
                else:
                    for k, key in enumerate(self.info_keys):
                        if k == 0: # dealing with raw data
                            self.data[key].append(self._preProcess(os.path.join(self.param['file_dir'], line[k])))
                        else:
                            self.data[key].append(line[k])
        hdf5 = h5py.File(dst_path, 'w')
        for k, key in enumerate(self.info_keys):
            if k == 0: # dealing with raw data
                dt = h5py.special_dtype(vlen=self.param['dtype'][k])
                hdf5.create_dataset(key, dtype=dt, data=np.array(self.data[key]))
            else:
                hdf5[key] = np.array(self.data[key]).astype(self.param['dtype'][k])
        hdf5.close()

    def generateFuel(self, dst_path):
        if dst_path.split('.')[-1] != 'hdf5':
            raise Exception('hdf5 file is recommended for storing compressed data.')
        duration = self._file2Byte(dst_path)
        print('+' + (59 * '-') + '+')
        print('| \033[1;35m%-23s\033[0m has been generated within \033[1;35m%6.3fs\033[0m |'
              % (os.path.basename(dst_path), duration))
        print('+' + (59 * '-') + '+')

    def editProperty(self, config=None, **kwargs):
        if config:
            kwargs = config
        for key in kwargs:
            if key not in self.modifiable_keys:
                print('%s is not a modifiable parameter or has not been defined.' % key)
            else:
                self.param[key] = kwargs[key]