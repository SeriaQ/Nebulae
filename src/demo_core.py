 #!/usr/bin/env python
'''
demo_dist
Created by Seria at 05/03/2021 11:01 PM

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
from nebulae import *

import numpy as np
import cv2
from time import time





def launch(cfg):
    ISIZE = cfg['hyper']['img_size']
    BSIZE = cfg['hyper']['batch_size']
    NEPOCH = cfg['hyper']['nepoch']
    LR = cfg['hyper']['lr']
    WD = cfg['hyper']['wd']
    DSTEP = cfg['hyper']['decay_step']
    DRATE = cfg['hyper']['decay_rate']
    WARM = cfg['hyper']['warmup']
    BRIGHT = cfg['aug']['bright']
    ROTATE = cfg['aug']['rotate']
    LROOT = cfg['env']['log_root']
    DROOT = cfg['env']['data_root']
    NGPU = cfg['env']['ngpu']
    TMODE = cfg['env']['train_mode']
    
    uv = power.Universe()
    kit.destine(121)
    # --------------------------------- Aerolog ---------------------------------- #
    def saveimg(stage, epoch, mile, mpe, value):
        if mile%32==0:
            cv2.imwrite('/root/proj/logs/ckpt/retro_%d_%d.png'%(epoch, mile), (value[:,:,0] * 255).astype(np.uint8))
    db = logs.DashBoard(log_dir=os.path.join(DROOT, "ckpt"),
                               window=15, divisor=15, span=70,
                               format={"Acc": [".2f", logs.PERCENT], "Loss": [".3f", logs.RAW]})#, 'Img': [saveimg, logs.INVIZ]})

    # --------------------------------- Cockpit ---------------------------------- #
    ng = power.Engine(device=power.GPU, ngpu=NGPU, multi_piston=TMODE=='dp', gearbox=neb.power.STATIC)
    tm = power.TimeMachine(save_dir=os.path.join(DROOT, "ckpt"),
                                 ckpt_dir=os.path.join(DROOT, "ckpt"))

    # ---------------------------------- Fuel ------------------------------------ #
    cb_train = fuel.Comburant(fuel.Random(0.5, fuel.Brighten(BRIGHT)),
                              fuel.Random(0.5, fuel.Rotate(ROTATE)),
                              fuel.Resize((ISIZE, ISIZE)),
                              fuel.End(),
                              fuel.HWC2CHW(),
                              fuel.Whiten(0.5, 0.5),
                              is_encoded=True)
    cb_dev = fuel.Comburant(fuel.Resize((ISIZE, ISIZE)),
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
    tkt = dp.mount(TrainSet(os.path.join(DROOT, "cifar10/cifar10_train.hdf5")),
                        batch_size=BSIZE, shuffle=True, nworker=4)
    tkd = dp.mount(DevSet(os.path.join(DROOT, "cifar10/cifar10_val.hdf5")),
                      batch_size=BSIZE, shuffle=False)

    # -------------------------------- Astro Dock --------------------------------- #
    class Net(nad.Craft):
        def __init__(self, nclass, scope):
            super(Net, self).__init__(scope)
            self.formulate(nac('res/in') >> nac('res/out'))

            # pad = dock.autopad((3, 3), 2)
            # self.conv = dock.Conv(3, 8, (3, 3), stride=2, padding=pad, b_init=dock.Void())
            # self.relu = dock.Relu()
            # pad = dock.autopad((2, 2), 2)
            # self.mpool = dock.MaxPool((2, 2), padding=pad)

            self.res = neb.astro.hangar.Resnet_V2_152((ISIZE, ISIZE, 3))
            self.flat = nad.Reshape()
            self.fc = nad.Dense(2048, nclass) # 512 2048

        def run(self, x):
            self['in'] = x
            # x = self.conv(x)
            # x = self.relu(x)
            # f = self.mpool(x)

            f = self.res(x)
            x = self.flat(f, (-1, 2048))
            y = self.fc(x)

            return y, f

    class Train(nad.Craft):
        def __init__(self, net, scope='TRAIN'):
            super(Train, self).__init__(scope)
            self.net = net
            self.loss = nad.SftmXE(is_one_hot=False)
            self.acc = nad.AccCls(multi_class=False, is_one_hot=False)
            self.optz = nad.Adam(self.net, LR, wd=WD, lr_decay=nad.StepLR(DSTEP, DRATE), warmup=WARM, )#mixp=True)

        @kit.Timer
        def run(self, x, z):
            self.net.off()
            with nad.Rudder() as rud:
                self.net.gear(rud)
                y, _ = self.net(x)
                loss = self.loss(y, z)
                acc = self.acc(y, z)
                loss, acc = uv.reduce((loss, acc))
                self.optz(loss)
            self.net.update()
            return loss, acc

    class Dev(nad.Craft):
        def __init__(self, net, scope='DEVELOP'):
            super(Dev, self).__init__(scope)
            self.net = net
            self.loss = nad.SftmXE(is_one_hot=False)
            self.acc = nad.AccCls(multi_class=False, is_one_hot=False)
            # self.retro = nad.Retroact()

        @kit.Timer
        def run(self, x, z):
            self.net.on()
            with nad.Nozzle() as noz:
                self.net.gear(noz)
                y, f = self.net(x)
                loss = self.loss(y, z)
                acc = self.acc(y, z)
                loss, acc = uv.reduce((loss, acc))
            return loss, acc#, m

    # --------------------------------- Inspect --------------------------------- #
    net = Net(10, 'cnn')
    # net.mixp()
    net = nad.EMA(net, on_device=True)
    net = net.gear(ng)
    net = uv.sync(net)
    train = Train(net)
    dev = Dev(net)
    sp = neb.logs.Inspector(export_path=os.path.join(LROOT, 'ckpt/res50'), verbose=True, onnx_ver=9)
    dummy_x = nad.coat(np.random.rand(1, 3, ISIZE, ISIZE).astype(np.float32))
    sp.dissect(net, dummy_x)
    sp.paint(net, dummy_x)

    # --------------------------------- Launcher --------------------------------- #
    gu = kit.GPUtil()
    gu.monitor()
    # tm.to(net, train.optz)
    best = 0
    for epoch in range(NEPOCH):
        mpe = dp.MPE[tkt]
        for mile in range(mpe):
            batch = dp.next(tkt)
            img, label = nad.coat(batch['image']), nad.coat(batch['label'])
            duration, loss, acc = train(img, label)
            loss = nad.shell(loss)
            acc = nad.shell(acc)
            probe = {'Acc': acc, 'Loss':loss}
            db(probe, epoch, mile, mpe, 'TRAIN', interval=2, \
                        is_global=True, is_elastic=True, in_loop=(0, 1), last_for=16)

        mpe = dp.MPE[tkd]
        for mile in range(mpe):
            batch = dp.next(tkd)
            # idx = int(dock.shell(batch['label'])[0])
            img, label = nad.coat(batch['image']), nad.coat(batch['label'])
            duration, loss, acc = dev(img, label)
            loss = nad.shell(loss)
            acc = nad.shell(acc)
            probe = {'Acc': acc, 'Loss': loss}#, 'Img': fm}
            db(probe, epoch, mile, mpe, 'DEV', interval=2, )#duration=duration)
        curr = db.read('Acc', 'DEV')
        if curr > best:
            tm.drop(net, train.optz)
            best = curr

        db.log(subdir='%03d'%epoch) #, history=os.path.join(LROOT, 'ckpt'))
    gu.status()



if __name__ == '__main__':
    # ----------------------------- Global Setting ------------------------------- #
    cfg = kit.parse_cfg('config_core.yml')
    if cfg['env']['train_mode'] == 'dt':
        mv = neb.power.Multiverse(launch, cfg['env']['ngpu'])
        mv(cfg)
    else:
        launch(cfg)