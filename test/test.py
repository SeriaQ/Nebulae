#!/usr/bin/env python
'''
test
Created by Seria at 05/11/2018 9:13 PM
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

import nebulae

fg = nebulae.toolkit.FuelGenerator(file_dir='/Users/Seria/Desktop/nebulae/test/mer',
                                  file_list='pae_val.csv',
                                  dst_path='/Users/Seria/Desktop/nebulae/test/mer/pae.hdf5',
                                  dtype=['uint8', 'int8'],
                                  channel=3,
                                  height=224,
                                  width=224,
                                  encode='png')
fg.propertyEdit(encode='jpeg')
# fg.generateFuel()

fd = nebulae.fuel.FuelDepot()
fd.loadFuel(name='test-img',
            batch_size=4,
            key_data='image',
            data_path='/Users/Seria/Desktop/nebulae/test/mer/pae.hdf5',
            # width=200, height=200,
            # resol_ratio=0.5,
            # spatial_aug='brightness,gamma_contrast',
            p_sa=(0.5, 0.5), theta_sa=(0.1, 1.2))
config = {'name':'test', 'batch_size':128}
fd.propertyEdit(dataname='test-img', config=config)
for s in range(fd.stepsPerEpoch('test')):
    batch = fd.nextBatch('test')
    print(fd.currentEpoch('test'), batch['label'])
fd.unloadFuel(name='test')

label = ['1 2','1 3', '0']
print(nebulae.toolkit.toOneHot(label, 5))