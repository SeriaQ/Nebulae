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
import nebulae as neb
from nebulae import kit, fuel
from nebulae.astro import dock, fn

import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import time




def launch(mv=None):
    kit.destine(121)
    size = 32
    # --------------------------------- Aerolog ---------------------------------- #
    def saveimg(stage, epoch, mile, mpe, value):
        if mile%32==0:
            plt.imsave('/root/proj/logs/ckpt/retro_%d_%d.jpg'%(epoch, mile), value[:,:,0], cmap='gray')
    db = neb.aerolog.DashBoard(log_dir="/root/proj/logs/ckpt",
                               window=15, divisor=15, span=70,
                               format={"Acc": [".2f", "percent"], "Loss": [".3f", "raw"]})#, 'Img': [saveimg, 'inviz']})

    # --------------------------------- Cockpit ---------------------------------- #
    ng = neb.cockpit.Engine(device=neb.cockpit.GPU, ngpu=2, multi_piston=True)# gearbox=neb.cockpit.FIXED)
    tm = neb.cockpit.TimeMachine(save_dir="/root/proj/logs/ckpt",
                                 ckpt_dir="/root/proj/logs/ckpt")

    # ---------------------------------- Fuel ------------------------------------ #
    cb_train = fuel.Comburant(fuel.Random(0.5, fuel.Brighten(0.1)),
                              fuel.Random(0.5, fuel.Rotate(10)),
                              fuel.Resize((size, size)),
                              fuel.End(),
                              fuel.HWC2CHW(),
                              fuel.Whiten(0.5, 0.5),
                              is_encoded=True)
    cb_dev = fuel.Comburant(fuel.Resize((size, size)),
                            fuel.End(),
                            fuel.HWC2CHW(),
                            fuel.Whiten(0.5, 0.5),
                            is_encoded=True)

    class TrainSet(fuel.Tank):
        def load(self, path):
            self.data = fuel.load_h5(path)
            return len(self.data['label'])

        # @kit.SPST
        def fetch(self, idx):
            ret = {}
            img = self.data['image'][idx]
            img = cb_train(img)
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
            img = cb_dev(img)
            ret['image'] = img
            label = self.data['label'][idx].astype('int64')
            ret['label'] = label
            return ret

    # {'image': 'vuint8', 'label': 'int64'}
    dp = fuel.Depot(ng)
    tkt = dp.mount(TrainSet("/root/proj/data/cifar10/cifar10_train.hdf5"),
                        batch_size=128, shuffle=True, nworker=4, prefetch=4)
    tkd = dp.mount(DevSet("/root/proj/data/cifar10/cifar10_val.hdf5"),
                      batch_size=32, shuffle=False)

    # -------------------------------- Space Dock --------------------------------- #
    class Net(dock.Craft):
        def __init__(self, nclass, scope):
            super(Net, self).__init__(scope)
            # self.formulate(fn('res/in') >> fn('res/out'))

            # pad = dock.autopad((3, 3), 2)
            # self.conv = dock.Conv(3, 8, (3, 3), stride=2, padding=pad, b_init=dock.Void())
            # self.relu = dock.Relu()
            # pad = dock.autopad((2, 2), 2)
            # self.mpool = dock.MaxPool((2, 2), padding=pad)

            self.res = neb.astro.hangar.Resnet_V2_152((size, size, 3))
            self.flat = dock.Reshape()
            self.fc = dock.Dense(2048, nclass) # 512 2048

        def run(self, x):
            self['in'] = x
            # x = self.conv(x)
            # x = self.relu(x)
            # f = self.mpool(x)

            f = self.res(x)
            x = self.flat(f, (-1, 2048))
            y = self.fc(x)

            return y, f

    class Train(dock.Craft):
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

    class Dev(dock.Craft):
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
    net = dock.EMA(Net(10, 'cnn'), on_device=True)
    net = net.gear(ng)
    train = Train(net)
    dev = Dev(net)
    sp = neb.aerolog.Inspector(export_path='/root/proj/logs/ckpt/res50', verbose=True, onnx_ver=9)
    dummy_x = dock.coat(np.random.rand(1, 3, size, size).astype(np.float32))
    sp.dissect(net, dummy_x)
    sp.paint(net, dummy_x)

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

        db.log() # history='/root/proj/logs/ckpt')



if __name__ == '__main__':
    # ----------------------------- Global Setting ------------------------------- #
    launch()