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

import nebulae as neb
from nebulae.fuel import depot
from nebulae.astrobase import dock

import numpy as np




def launch(mv=None):
    mv.init()

    # --------------------------------- Aerolog ---------------------------------- #
    bp = neb.aerolog.BluePrint(hidden=["is_train"], verbose=True)
    db = neb.aerolog.DashBoard(log_path="/users/seria/data/cifar100/ckpt",
                               window=15, divisor=15, span=70,
                               format={"Acc": [".2f", "percent"], "Loss": [".3f", "raw"]})

    # --------------------------------- Cockpit ---------------------------------- #
    ng = neb.cockpit.Engine(device="gpu", ngpus=mv.nworld)
    tm = neb.cockpit.TimeMachine(save_path="/users/seria/data/cifar100/ckpt",
                                 ckpt_path="/users/seria/data/cifar100/ckpt")

    # ---------------------------------- Fuel ------------------------------------ #
    cb_train = depot.Comburant(depot.Resize((112, 112)),
                               depot.Crop((96, 96)),
                               depot.Flip(0.5, depot.VERTICAL),
                               is_encoded=True)
    cb_dev = depot.Comburant(depot.Resize((96, 96)),
                             is_encoded=True)

    def fetcher_train(data, idx):
        ret = {}
        img = cb_train(data['image'][idx])
        img = np.transpose(img, (2, 0, 1)).astype('float32')
        ret['image'] = img
        label = data['label'][idx].astype('int64')
        ret['label'] = label
        return ret

    def fetcher_dev(data, idx):
        ret = {}
        img = cb_dev(data['image'][idx])
        img = np.transpose(img, (2, 0, 1)).astype('float32')
        ret['image'] = img
        label = data['label'][idx].astype('int64')
        ret['label'] = label
        return ret

    tk_train = depot.Tank("/users/seria/data/cifar100/cifar100_train.hdf5", 'image',
                       batch_size=128, shuffle=True, fetch_fn=fetcher_train)
    tk_dev = depot.Tank("/users/seria/data/cifar100/cifar100_val.hdf5", 'image',
                       batch_size=32, shuffle=False, fetch_fn=fetcher_dev)

    # -------------------------------- Space Dock --------------------------------- #
    with mv.scope():
        class Net(dock.Craft):
            def __init__(self, scope, nclass):
                super(Net, self).__init__(scope)
                pad = dock.autoPad((96, 96), (3, 3),  2)
                self.conv = dock.Conv(3, 8, (3, 3), stride=2, padding=pad, b_init=None)
                self.relu = dock.Relu()
                pad = dock.autoPad((56, 56), (2, 2), 2)
                self.mpool = dock.MaxPool((2, 2), padding=pad)
                self.flat = dock.Reshape()
                self.fc = dock.Dense(4608, nclass)

            def run(self, x):
                bs = x.shape[0]
                self['input'] = x
                self['conv'] = self.conv(self['input'])
                self['relu'] = self.relu(self['conv'])
                self['mpool'] = self.mpool(self['relu'])
                x = self.flat(self['mpool'], (bs, -1))
                self['out'] = self.fc(x)

                return self['out']

        class Train(dock.Craft):
            def __init__(self, net, scope='TRAIN'):
                super(Train, self).__init__(scope)
                self.scope = scope
                self.net = net
                self.loss = dock.SftmXE(is_one_hot=False)
                self.acc = dock.AccCls(multi_class=False, is_one_hot=False)
                self.optz = dock.Momentum(net, 2e-3, wd=4e-5, lr_decay=dock.STEP, lr_params=(300, 0.8))

            @neb.toolkit.Timer
            @mv.Executor
            def run(self, x, z):
                with dock.Rudder() as rud:
                    self.net.gear(rud)
                    y = self.net(x)
                    loss = self.loss(y, z)
                    loss = mv.reduce(loss)
                    acc = self.acc(y, z)
                    acc = mv.reduce(acc)
                self.optz(loss)
                return loss, acc

        class Dev(dock.Craft):
            def __init__(self, net, scope='DEVELOP'):
                super(Dev, self).__init__(scope)
                self.scope = scope
                self.net = net
                self.loss = dock.SftmXE(is_one_hot=False)
                self.acc = dock.AccCls(multi_class=False, is_one_hot=False)

            @neb.toolkit.Timer
            @mv.Executor
            def run(self, x, z):
                with dock.Nozzle() as noz:
                    self.net.gear(noz)
                    y = self.net(x)
                    loss = self.loss(y, z)
                    loss = mv.reduce(loss)
                    acc = self.acc(y, z)
                    acc = mv.reduce(acc)
                return loss, acc

    # --------------------------------- Launcher --------------------------------- #
    net = Net('cnn', 100)
    net.gear(ng)
    net, tk_train, tk_dev = mv.sync(net, (tk_train, tk_dev))
    train = Train(net)
    dev = Dev(net)

    best = 0
    for epoch in range(10):
        mpe = tk_train.MPE
        for mile in range(mpe):
            batch = tk_train.next()
            img, label = ng.coat(batch['image']), ng.coat(batch['label'])
            duration, loss, acc = train(img, label)
            loss = ng.shell(loss)
            acc = ng.shell(acc)
            probe = {'Acc': acc, 'Loss':loss}
            db.gauge(probe, mile, epoch, mpe, 'TRAIN', interval=5, duration=duration)

        mpe = tk_dev.MPE
        for mile in range(mpe):
            batch = tk_dev.next()
            img, label = ng.coat(batch['image']), ng.coat(batch['label'])
            duration, loss, acc = dev(img, label)
            loss = ng.shell(loss)
            acc = ng.shell(acc)
            probe = {'Acc': acc, 'Loss': loss}
            db.gauge(probe, mile, epoch, mpe, 'DEV', interval=1, duration=duration)
        curr = db.read('Acc', 'DEV')
        if curr is not None and curr > best:
            tm.drop(net)
            best = curr

    db.log()



if __name__ == '__main__':
    # ----------------------------- Global Setting ------------------------------- #
    mv = neb.law.Multiverse(launch, 4)
    mv()
