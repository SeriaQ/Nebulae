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
import tensorflow as tf

# fg = nebulae.toolkit.FuelGenerator(file_dir='/Users/Seria/Desktop/nebulae/test/mer',
#                                   file_list='pae_val.csv',
#                                   dtype=['uint8', 'int8'],
#                                   channel=3,
#                                   height=224,
#                                   width=224,
#                                   encode='png')
# fg.editProperty(encode='jpeg')
# fg.generateFuel('/Users/Seria/Desktop/nebulae/test/mer/pae.hdf5')

# fd = nebulae.fuel.FuelDepot()
# fd.loadFuel(name='test-img',
#             batch_size=4,
#             key_data='image',
#             data_path='/Users/Seria/Desktop/nebulae/test/mer/pae.hdf5',
#             width=200, height=200,
#             resol_ratio=0.5,
#             spatial_aug='brightness,gamma_contrast',
#             p_sa=(0.5, 0.5), theta_sa=(0.1, 1.2))
# config = {'name':'test', 'batch_size':128}
# fd.editProperty(dataname='test-img', config=config)
# for s in range(fd.stepsPerEpoch('test')):
#     batch = fd.nextBatch('test')
#     print(fd.currentEpoch('test'), batch['label'].shape)
# fd.unloadFuel('test')
#
# label = ['1 2','1 3', '0']
# print(nebulae.toolkit.toOneHot(label, 5))



# fg = nebulae.toolkit.FuelGenerator(file_dir='/Users/Seria/Desktop/Luck/Competition/Workshop/plane/blob_train_image_data',
#                                   file_list='por_val.csv',
#                                   dtype=['uint8', 'int64'],
#                                   channel=1,
#                                   height=32,
#                                   width=32,
#                                   encode='png')
# fg.generateFuel('/Users/Seria/Desktop/nebulae/test/porosity_val.hdf5')
fd = nebulae.fuel.FuelDepot()
fname = 'por'
fd.loadFuel(name=fname,
            batch_size=32,
            channel=1,
            data_path='/Users/Seria/Desktop/nebulae/test/porosity_train.hdf5',
            data_key='image',
            spatial_aug='flip',
            p_sa=(0.5,), theta_sa=(0,))

ls = nebulae.aerolog.LayoutSheet('/Users/Seria/Desktop/nebulae/test/por_ls')

COMP = nebulae.spacedock.Component()
scope = 'sc'
sc = nebulae.spacedock.SpaceCraft(scope, layout_sheet=ls)
sc.fuelLine('input', (None, 32, 32, 1), 'float32')
sc.fuelLine('label', (None), 'int64')
comp_conv1 = (COMP.CONV(name='conv1', input=sc.layout['input'], kernel_size=[3, 3], in_out_chs=[1, 4])
            >> COMP.RELU(name='relu1')
            >> COMP.MAX_POOL(name='max_pool1'))
comp_conv2 = (COMP.CONV(name='conv2', input=sc.layout['input'], kernel_size=[3, 3], in_out_chs=[4, 8])
            >> COMP.RELU(name='relu2'))
            # >> COMP.MAX_POOL(name='max_pool2'))
comp_conv3 = (COMP.CONV(name='conv3', input=sc.layout['input'], kernel_size=[3, 3], in_out_chs=[1, 8])
            >> COMP.RELU(name='relu3')
            >> COMP.MAX_POOL(name='max_pool3'))
outnode = sc.assembleComp((comp_conv1 >> comp_conv2) & comp_conv3)

comp_out = COMP.FLAT(name='flat1', input=sc.layout[outnode]) \
            >> COMP.DENSE(name='fc1', out_chs=1) \
            >> COMP.SIGMOID(name='sftm1')
outnode = sc.assembleComp(comp_out)
ls._generateLS()


label = tf.cast(sc.layout['label'], tf.float32)
prob = sc.layout[outnode]
cost_obj = tf.reduce_mean(-label*tf.log(prob)-(1-label)*tf.log(1-prob))
# cost_obj = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(sc.layout['label'], prob))
cost_reg = 4e-5 * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))
cost = cost_obj + cost_reg

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
global_step = tf.Variable(0, trainable=False)
decayed_lr = tf.train.exponential_decay(0.01, global_step, 1000, 0.8, staircase=True)
optz = tf.train.MomentumOptimizer(decayed_lr, 0.9, use_nesterov=True)
# optz = tf.train.AdamOptimizer(decayed_lr)
grad_var_pairs = optz.compute_gradients(cost,
                                        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
capped_gvs = [(tf.clip_by_value(grad, -4., 4.), var) for grad, var in grad_var_pairs]
train_op = optz.apply_gradients(capped_gvs, global_step)

correct = tf.equal(tf.cast(tf.round(prob), tf.int64), sc.layout['label'])
# correct = tf.equal(tf.argmax(prob, 1), sc.layout['label'])
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
sess.run(tf.global_variables_initializer())

for s in range(fd.stepsPerEpoch(fname)):
    batch = fd.nextBatch(fname)
    _, acc, loss, pr = sess.run([train_op, accuracy, cost, prob], feed_dict={sc.layout['input']: batch['image'],
                                                                   sc.layout['label']: batch['label']})
    # print(pr)
    # print(batch['label'])
    # import pdb
    # pdb.set_trace()
    if s % 64 == 0:
        print('epoch #%d: acc: %.2f%%, loss: %.4f' % (fd.currentEpoch(fname), acc * 100, loss))

fd.unloadFuel(fname)