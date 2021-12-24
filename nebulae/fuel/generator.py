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
from piexif import remove as rm_exif

from ..toolkit.decorator import Timer
from ..law import Constant

class Generator(object):

    def __init__(self, config=None, file_dir=None, file_list=None, dtype=None, is_seq=False):
        self.param = {}
        self.modifiable_keys = ['file_dir', 'file_list', 'dtype']
        if config:
            self.param['file_dir'] = config.get('file_dir')
            self.param['file_list'] = config.get('file_list')
            self.param['dtype'] = config.get('dtype')
            self.param['is_seq'] = config.get('is_seq', is_seq)
        else:
            self.param['file_dir'] = file_dir
            self.param['file_list'] = file_list
            self.param['dtype'] = dtype
            self.param['is_seq'] = is_seq

        # check if key arguments are valid
        if self.param['file_list'].split('.')[-1] != 'csv':
            raise Exception('NEBULAE ERROR ⨷ file list should be a csv file.')
        for dt in self.param['dtype']:
            if dt.strip('v') not in Constant.VALID_DTYPE:
                raise Exception('NEBULAE ERROR ⨷ %s is not a valid data type.' % dt)

    def _compress(self, img_path, height, width, channel, encode, keep_exif):
        ch_err = Exception('NEBULAE ERROR ⨷ images having %d channels are invalid.' % channel)
        with io.BytesIO() as buffer:
            if keep_exif:
                if channel == 1:
                    image = Image.open(img_path).convert('L')
                elif channel == 3:
                    image = Image.open(img_path).convert('RGB')
                else:
                    raise ch_err
                if width>0 and height>0:
                    image = image.resize((width, height))
                image.save(buffer, format=encode)
            else:
                rm_exif(img_path, buffer)
            encoded_bytes = buffer.getvalue()
        np_bytes = np.frombuffer(encoded_bytes, dtype=np.uint8)

        return np_bytes

    def _write(self, dst_path, patch, data, info_keys, max_frames):
        if patch < 0:
            hdf5 = h5py.File(dst_path, 'w')
        else:
            hdf5 = h5py.File(dst_path[:-5] + '_%d.hdf5'%patch, 'w')
        for k, key in enumerate(info_keys):
            if isinstance(data[key], list):
                data[key] = np.array(data[key], dtype='object')
            dt = self.param['dtype'][k]
            if dt.startswith('v') or dt=='str':
                if dt=='str':
                    dt = str
                else:
                    dt = dt[1:]
                sdt = h5py.special_dtype(vlen=dt)
                hdf5.create_dataset(key, dtype=sdt, data=data[key])
            else:
                hdf5[key] = data[key].astype(dt)
        if self.param['is_seq']:
            hdf5[Constant.FRAME_KEY] = max_frames
        hdf5.close()

    @Timer
    def _file2Byte(self, dst_path, height, width, channel, encode, shards, keep_exif):
        data = {}
        print('+' + (80 * '-') + '+')
        nsample = len(open(os.path.join(self.param['file_dir'], self.param['file_list']), 'r').readlines()) - 1
        patch_size = int(nsample/shards) + 1
        if shards == 1:
            patch = -1
        else:
            patch = 0
        with open(os.path.join(self.param['file_dir'], self.param['file_list']), 'r') as filelist:
            content = csv.reader(filelist, delimiter=Constant.CHAR_SEP, quotechar=Constant.FIELD_SEP)
            ending_char = '\r'
            for l, line in enumerate(content):
                # display progress bar
                progress = int(l/nsample*39 + 0.4)
                yellow_bar = progress * ' '
                space_bar = (39-progress) * ' '
                if l == nsample:
                    ending_char = '\n'
                print('| Integrating data %7d / %-7d ⊰⟦\033[43m%s\033[0m%s⟧⊱ |'
                      % (l, nsample, yellow_bar, space_bar), end=ending_char)
                if l == 0: # initialize data dict
                    info_keys = line
                    if len(line) != len(self.param['dtype']):
                        raise Exception('NEBULAE ERROR ⨷ number of given dtypes does not match the provided csv file.')
                    for key in line:
                        data[key] = []
                    max_frames = 0
                else:
                    for k, key in enumerate(info_keys):
                        if k == 0 and len(encode) > 0: # dealing with raw data
                            if self.param['is_seq']:
                                csl = line[k].split(Constant.CHAR_SEP) # comma separated line
                                max_frames = len(csl) if len(csl) > max_frames else max_frames
                                temp_data = []
                                for f in csl:
                                    temp_data.append(self._compress(os.path.join(self.param['file_dir'], f),
                                                                    height, width, channel, encode, keep_exif))
                                data[key].append(np.array(temp_data).astype(self.param['dtype'][k].strip('v')))
                            else:
                                data[key].append(self._compress(os.path.join(self.param['file_dir'], line[k]),
                                                                height, width, channel, encode, keep_exif))
                        else:
                            csl = line[k].split(Constant.CHAR_SEP) # comma separated line
                            if len(csl) == 1 and not self.param['dtype'][k].startswith('v'):
                                data[key].append(line[k])
                            else:
                                temp_data = []
                                for f in csl:
                                    temp_data.append(f)
                                data[key].append(np.array(temp_data).astype(self.param['dtype'][k].strip('v')))
                    if l % patch_size == 0:
                        self._write(dst_path, patch, data, info_keys, max_frames)
                        for key in info_keys:
                            data[key] = []
                        max_frames = 0
                        patch += 1
        self._write(dst_path, patch, data, info_keys, max_frames)

    def generate(self, dst_path, height=0, width=0, channel=3, encode='JPEG', shards=1, keep_exif=True):
        if not (h5py.is_hdf5(dst_path) or dst_path.split('.')[-1]=='hdf5'):
            raise Exception('NEBULAE ERROR ⨷ hdf5 file is recommended for storing compressed data.')
        if shards < 1 or (not isinstance(shards, int)):
            raise ValueError('NEBULAE ERROR ⨷ the number of shards must be an positive integer.')
        duration = self._file2Byte(dst_path, height, width, channel, encode, shards, keep_exif)[0]
        print('+' + (80 * '-') + '+')
        print('| \033[1;35m%-38s\033[0m has been generated within \033[1;35m%12.2fs\033[0m |'
              % (os.path.basename(dst_path), duration))
        print('+' + (80 * '-') + '+')

    def modify(self, config=None, **kwargs):
        if config:
            kwargs = config
        for key in kwargs:
            if key not in self.modifiable_keys:
                raise KeyError('NEBULAE ERROR ⨷ %s is not a modifiable parameter or has not been defined.' % key)
            else:
                self.param[key] = kwargs[key]