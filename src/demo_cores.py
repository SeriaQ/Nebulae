 #!/usr/bin/env python
'''
demo_cores
Created by Seria at 05/11/2018 9:13 PM

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

'''
This is a demo script to demonstrate how to build a simple neural network for classification
with different backend cores. Training and validation are included as well.
'''

import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import nebulae as neb
from nebulae import kit, fuel, astro
from nebulae.astro import dock, hangar

import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import time





def launch(mv=None):
    kit.destine(121)
    # --------------------------------- Aerolog ---------------------------------- #
    def saveimg(stage, epoch, mile, mpe, value):
        if mile%32==0:
            plt.imsave('/Users/Seria/Desktop/nebulae/test/ckpt/retro_%d_%d.jpg'%(epoch, mile), value[:,:,0], cmap='gray')
    db = neb.aerolog.DashBoard(log_path="/Users/Seria/Desktop/nebulae/test/ckpt",
                               window=15, divisor=15, span=70,
                               format={"Acc": [".2f", "percent"], "Loss": [".3f", "raw"]})#, 'Img': [saveimg, 'inviz']})

    # --------------------------------- Cockpit ---------------------------------- #
    ng = neb.cockpit.Engine(device=neb.cockpit.CPU)
    tm = neb.cockpit.TimeMachine(save_path="/Users/Seria/Desktop/nebulae/test/ckpt",
                                 ckpt_path="/Users/Seria/Desktop/nebulae/test/ckpt")

    # ---------------------------------- Fuel ------------------------------------ #
    cb_train = fuel.Comburant(fuel.Random(0.5, fuel.Brighten(0.1)),
                              fuel.Random(0.5, fuel.Rotate(10)),
                              fuel.Resize((32, 32)),
                              fuel.HWC2CHW(),
                              is_encoded=True)
    cb_dev = fuel.Comburant(fuel.Resize((32, 32)),
                            fuel.HWC2CHW(),
                            is_encoded=True)

    class TrainSet(fuel.Tank):
        def load(self, path):
            self.data = fuel.load_h5(path)
            return len(self.data['label'])

        # @kit.SPST
        def fetch(self, idx):
            ret = {}
            img = self.data['image'][idx]
            img = cb_train(img) * 2 - 1
            ret['image'] = img
            label = self.data['label'][idx].astype('int64')
            ret['label'] = label
            return ret


    class DevSet(fuel.Tank):
        def load(self, path):
            self.data = fuel.load_h5(path)
            return len(self.data['label'])

        # @kit.SPST
        def fetch(self, idx):
            ret = {}
            img = self.data['image'][idx]
            img = cb_dev(img) * 2 - 1
            ret['image'] = img
            label = self.data['label'][idx].astype('int64')
            ret['label'] = label
            return ret

    # {'image': 'vuint8', 'label': 'int64'}
    dp = fuel.Depot(ng)
    tkt = dp.mount(TrainSet("/Users/Seria/Desktop/nebulae/test/data/cifar10/cifar10_train.hdf5"),
                        batch_size=128, shuffle=True, nworker=2)
    tkd = dp.mount(DevSet("/Users/Seria/Desktop/nebulae/test/data/cifar10/cifar10_val.hdf5"),
                      batch_size=32, shuffle=False)

    # -------------------------------- Space Dock --------------------------------- #
    class Net(astro.Craft):
        def __init__(self, nclass, scope):
            super(Net, self).__init__(scope)
            # pad = dock.autopad((3, 3), 2)
            # self.conv = dock.Conv(3, 8, (3, 3), stride=2, padding=pad, b_init=dock.Void())
            # self.relu = dock.Relu()
            # pad = dock.autopad((2, 2), 2)
            # self.mpool = dock.MaxPool((2, 2), padding=pad)
            self.res = hangar.Resnet_V2_50((32, 32, 3))
            self.flat = dock.Reshape()
            self.fc = dock.Dense(2048, nclass) # 512 2048

        def run(self, x):
            self['input'] = x

            # x = self.conv(self['input'])
            # x = self.relu(x)
            # self['feat'] = self.mpool(x)

            self['feat'] = self.res(self['input'])
            x = self.flat(self['feat'], (-1, 2048))
            self['out'] = self.fc(x)

            return self['out'], self['feat']

    class Train(astro.Craft):
        def __init__(self, net, scope='TRAIN'):
            super(Train, self).__init__(scope)
            self.net = net
            self.loss = dock.SftmXE(is_one_hot=False)
            self.acc = dock.AccCls(multi_class=False, is_one_hot=False)
            self.optz = dock.Adam(self.net, 3e-3, wd=0, lr_decay=dock.StepLR(300, 0.8), warmup=390)

        @kit.Timer
        def run(self, x, z, de=False):
            self.net.off()
            with dock.Rudder() as rud:
                self.net.gear(rud)
                y, _ = self.net(x)
                loss = self.loss(y, z)
                acc = self.acc(y, z)
                self.optz(loss)
            self.net.update()
            return loss, acc

    class Dev(astro.Craft):
        def __init__(self, net, scope='DEVELOP'):
            super(Dev, self).__init__(scope)
            self.net = net
            self.loss = dock.SftmXE(is_one_hot=False)
            self.acc = dock.AccCls(multi_class=False, is_one_hot=False)
            # self.retro = dock.Retroact()

        @kit.Timer
        def run(self, x, z, idx):
            self.net.on()
            with dock.Nozzle() as noz:
                self.net.gear(noz)
                y, f = self.net(x)
                loss = self.loss(y, z)
                acc = self.acc(y, z)
                # m = self.retro(f, y[0, idx])[0]
            return loss, acc#, m

    # --------------------------------- Inspect --------------------------------- #
    # net = Net(10, 'cnn')
    net = dock.EMA(Net(10, 'cnn'))
    net = net.gear(ng)
    train = Train(net)
    dev = Dev(net)
    sp = neb.aerolog.Inspector(verbose=True)
    dummy_x = dock.coat(np.random.rand(1,3,32,32).astype(np.float32))
    sp.dissect(net, dummy_x)

    # --------------------------------- Launcher --------------------------------- #
    # tm.to(net, train.optz)
    best = 0
    for epoch in range(5):
        mpe = dp.MPE[tkt]
        for mile in range(mpe):
            batch = dp.next(tkt)
            img, label = dock.coat(batch['image']), dock.coat(batch['label'])
            duration, loss, acc = train(img, label, epoch)
            loss = dock.shell(loss)
            acc = dock.shell(acc)
            probe = {'Acc': acc, 'Loss':loss}
            db.gauge(probe, mile, epoch, mpe, 'TRAIN', interval=2, duration=duration, is_global=True, is_elastic=True)

        mpe = dp.MPE[tkd]
        for mile in range(mpe):
            batch = dp.next(tkd)
            idx = int(dock.shell(batch['label'])[0])
            img, label = dock.coat(batch['image']), dock.coat(batch['label'])
            duration, loss, acc = dev(img, label, idx)
            loss = dock.shell(loss)
            acc = dock.shell(acc)
            # fm = dock.shell(fm)
            probe = {'Acc': acc, 'Loss': loss}#, 'Img': fm}
            db.gauge(probe, mile, epoch, mpe, 'DEV', interval=2, duration=duration)
        curr = db.read('Acc', 'DEV')
        if curr > best:
            tm.drop(net, train.optz)
            best = curr

    db.log() # history='/Users/Seria/Desktop/nebulae/test/ckpt')



if __name__ == '__main__':
    # ----------------------------- Global Setting ------------------------------- #
    launch()