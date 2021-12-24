 #!/usr/bin/env python
'''
demo_cores
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

'''
This is a demo script to demonstrate how to build a simple neural network for classification
with different backend cores. Training and validation are included as well.
'''

import os
os.environ['NEB_CORE'] = 'pytorch'
#os.environ['NEB_CORE'] = 'tensorflow'

import nebulae as neb
from nebulae.fuel import depot
from nebulae.astrobase import dock, hangar

import numpy as np
import matplotlib.pyplot as plt
from time import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



def launch(mv=None):
    # --------------------------------- Aerolog ---------------------------------- #
    bp = neb.aerolog.BluePrint(hidden=["is_train"], verbose=True)

    def saveimg(stage, epoch, mile, mpe, value):
        if mile%32==0:
            plt.imsave('/Users/Seria/Desktop/nebulae/test/ckpt/retro_%d_%d.jpg'%(epoch, mile), value[:,:,0], cmap='gray')
    db = neb.aerolog.DashBoard(log_path="/Users/Seria/Desktop/nebulae/test/ckpt",
                               window=15, divisor=15, span=70,
                               format={"Acc": [".2f", "percent"], "Loss": [".3f", "raw"]})#, 'Img': [saveimg, 'inviz']})

    # --------------------------------- Cockpit ---------------------------------- #
    ng = neb.cockpit.Engine(device="cpu")
    tm = neb.cockpit.TimeMachine(save_path="/Users/Seria/Desktop/nebulae/test/ckpt",
                                 ckpt_path="/Users/Seria/Desktop/nebulae/test/ckpt")

    # ---------------------------------- Fuel ------------------------------------ #
    cb_train = depot.Comburant(depot.Brighten(0.1),
                               depot.Rotate(10),
                               depot.Resize((32, 32)),
                               depot.HWC2CHW(),
                               is_encoded=True)#depot.HWC2CHW(),
    cb_dev = depot.Comburant(depot.Resize((32, 32)),
                             depot.HWC2CHW(),
                             is_encoded=True)

    def fetcher_train(data, idx):
        ret = {}
        img = data['image'][idx]
        ret['image'] = img
        label = data['label'][idx].astype('int64')
        ret['label'] = label
        return ret

    def prep_train(data):
        data['image'] = cb_train(data['image'])*2 - 1
        # data['image'] = np.transpose(data['image'], (2, 0, 1)).astype('float32')
        return data

    def fetcher_dev(data, idx):
        ret = {}
        img = data['image'][idx]
        ret['image'] = img
        label = data['label'][idx].astype('int64')
        ret['label'] = label
        return ret

    def prep_dev(data):
        data['image'] = cb_dev(data['image'])*2 - 1
        # data['image'] = np.transpose(data['image'], (2, 0, 1)).astype('float32')
        return data

    tk_train = depot.Tank("/Users/Seria/Desktop/nebulae/test/data/cifar10/cifar10_train.hdf5",
                          {'image': 'vunit8', 'label': 'int64'},
                          batch_size=128, shuffle=True, fetch_fn=fetcher_train, prep_fn=prep_train)
    tk_dev = depot.Tank("/Users/Seria/Desktop/nebulae/test/data/cifar10/cifar10_val.hdf5",
                        {'image': 'vuint8', 'label': 'int64'},
                        batch_size=32, shuffle=False, fetch_fn=fetcher_dev, prep_fn=prep_dev)

    # -------------------------------- Space Dock --------------------------------- #
    class Net(dock.Craft):
        def __init__(self, nclass, scope):
            super(Net, self).__init__(scope)
            pad = dock.autoPad((32, 32), (3, 3),  2)
            self.conv = dock.Conv(3, 8, (3, 3), stride=2, padding=pad, b_init=dock.Void())
            self.relu = dock.Relu()
            pad = dock.autoPad((16, 16), (2, 2), 2)
            self.mpool = dock.MaxPool((2, 2), padding=pad)
            # self.res = hangar.VGG_16((32, 32, 3), engine=ng)
            self.flat = dock.Reshape()
            self.fc = dock.Dense(512, nclass) #4096

        def run(self, x):
            bs = x.shape[0]
            self['input'] = x
            x = self.conv(self['input'])
            x = self.relu(x)
            self['feat'] = self.mpool(x)

            # self['feat'] = self.res(self['input'])
            x = self.flat(self['feat'], (bs, -1))
            self['out'] = self.fc(x)

            return self['out'], self['feat']

    class Train(dock.Craft):
        def __init__(self, net, scope='TRAIN'):
            super(Train, self).__init__(scope)
            self.net = net
            self.loss = dock.SftmXE(is_one_hot=False)
            self.acc = dock.AccCls(multi_class=False, is_one_hot=False)
            self.optz = dock.Momentum(self.net, 3e-3, wd=4e-5, lr_decay=dock.StepLR(300, 0.8), warmup=300)

        @neb.toolkit.Timer
        def run(self, x, z):
            # if self.net.swapped:
            #     self.net.swap()
            with dock.Rudder() as rud:
                self.net.gear(rud)
                y, _ = self.net(x)
                loss = self.loss(y, z)
                acc = self.acc(y, z)
                self.optz(loss)
            # self.net.update()
            return loss, acc

    class Dev(dock.Craft):
        def __init__(self, net, scope='DEVELOP'):
            super(Dev, self).__init__(scope)
            self.net = net
            self.loss = dock.SftmXE(is_one_hot=False)
            self.acc = dock.AccCls(multi_class=False, is_one_hot=False)
            # self.retro = dock.Retroact()

        @neb.toolkit.Timer
        def run(self, x, z, idx):
            # if not self.net.swapped:
            #     self.net.swap()
            with dock.Nozzle() as noz:
                self.net.gear(noz)
                y, f = self.net(x)
                loss = self.loss(y, z)
                acc = self.acc(y, z)
                # print(tf.gradients(y[0, idx], f))
                # print(noz.gradient(y[0, idx], f))
                # m = self.retro(f, y[0, idx], noz)[0]
            return loss, acc#, m

    # --------------------------------- Launcher --------------------------------- #
    net = Net(10, 'cnn')
    # net = dock.EMA(Net(10, 'cnn'))
    net.gear(ng)
    train = Train(net)
    dev = Dev(net)

    # with neb.aerolog.CtrlPanel(db) as cp:
    best = 0
    for epoch in range(10):
        mpe = tk_train.MPE
        for mile in range(mpe):
            batch = tk_train.next()
            img, label = dock.coat(batch['image']), dock.coat(batch['label'])
            duration, loss, acc = train(img, label)
            loss = dock.shell(loss)
            acc = dock.shell(acc)
            probe = {'Acc': acc, 'Loss':loss}
            db.gauge(probe, mile, epoch, mpe, 'TRAIN', interval=10, duration=duration)

        mpe = tk_dev.MPE
        for mile in range(mpe):
            batch = tk_dev.next()
            idx = int(dock.shell(batch['label'])[0])
            # import pdb; pdb.set_trace()
            img, label = dock.coat(batch['image']), dock.coat(batch['label'])
            duration, loss, acc = dev(img, label, idx)
            loss = dock.shell(loss)
            acc = dock.shell(acc)
            # fm = dock.shell(fm)
            probe = {'Acc': acc, 'Loss': loss}#, 'Img': fm}
            db.gauge(probe, mile, epoch, mpe, 'DEV', interval=2, duration=duration)
        curr = db.read('Acc', 'DEV')
        if curr is not None and curr > best:
            tm.drop(net)
            best = curr

            # cp.refresh()
        db.log()
        # cp.actuate()



if __name__ == '__main__':
    # ----------------------------- Global Setting ------------------------------- #
    launch()