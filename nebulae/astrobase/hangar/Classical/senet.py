#!/usr/bin/env python
'''
garage
Created by Seria at 03/01/2019 8:32 PM
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
from ..component import Component

def _residual(comp, inputs, shallow, deep, stride, name):
    COMP = Component(comp.channel_major, comp.time_major)
    inputs, is_train = inputs
    res_block = (COMP.CONV(name=name + '_conv_pre', inputs=inputs, kernel=[1, 1],
                          stride=[stride, stride], out_chs=shallow)
                >> COMP.BATCH_NORM(name=name + '_bn_mid', is_train=is_train)
                >> COMP.RELU(name=name + '_relu_mid')
                >> COMP.CONV(name=name + '_conv_mid', kernel=[3, 3], out_chs=shallow)
                >> COMP.BATCH_NORM(name=name + '_bn_post', is_train=is_train)
                >> COMP.RELU(name=name + '_relu_post')
                >> COMP.CONV(name=name + '_conv_post', kernel=[1, 1], out_chs=deep))
    return res_block

def _bottleNeckSE(sc, comp, inputs, shallow, deep, scope, name, down_sample=False, pre_activate=True):
    COMP = Component(comp.channel_major, comp.time_major)
    inputs, is_train = inputs
    if pre_activate:
        inputs = sc.assemble(COMP.BATCH_NORM(name=name + '_bn_pre',
                                            inputs=inputs, is_train=is_train)
                            >> COMP.RELU(name=name + '_relu_pre'), sub_scope=scope)
    if down_sample:
        stride = 2
        idt_path = (COMP.CONV(name=name + '_conv_idt', inputs=inputs,
                              kernel=[1, 1], stride=[2, 2], out_chs=deep))
    else:
        stride = 1
        if pre_activate:
            idt_path = (COMP.DUPLICATE(name=COMP.rmscope(inputs), inputs=inputs))
        else:
            idt_path = (COMP.CONV(name=name + '_conv_idt', inputs=inputs, kernel=[1, 1], out_chs=deep))

    res_path = sc.assemble(_residual(comp, (inputs, is_train),
                                           shallow, deep, stride, name), sub_scope=scope)
    se_path = (COMP.DUPLICATE(name=COMP.rmscope(res_path), inputs=res_path)
               * _squeezeExicitation(comp, res_path, shallow//4, deep, name))

    return sc.assemble(idt_path + se_path, sub_scope=scope)

def _squeezeExicitation(comp, inputs, shallow, deep, name):
    COMP = Component(comp.channel_major, comp.time_major)
    se_block = (COMP.AVG_POOL(name=name+'_se_avg_pool', inputs=inputs, keep_size=False, if_global=True)
                >> COMP.CONV(name=name+'_se_down', kernel=[1,1], out_chs=shallow)
                >> COMP.RELU(name=name+'_se_relu')
                >> COMP.CONV(name=name+'_se_up', kernel=[1,1], out_chs=deep)
                >> COMP.SIGMOID(name=name+'_se_sigm'))
    return se_block

def _se_resnet_v2(sc, comp, inputs, is_train, scope, blocks):
    COMP = Component(comp.channel_major, comp.time_major)
    net = sc.assemble(COMP.CONV(name='conv0', inputs=inputs,
                                kernel=[7, 7], stride=[2, 2], out_chs=64)
             >> COMP.MAX_POOL(name='max_pool0', kernel=[3, 3]), sub_scope=scope)
    # building block 1
    net = _bottleNeckSE(sc, comp, [net, is_train], 64, 256, scope, pre_activate=False, name='block1_0')
    for l in range(1, blocks[0]):
        net = _bottleNeckSE(sc, comp, [net, is_train], 64, 256, scope, name='block1_'+str(l))
    # building block 2
    net = _bottleNeckSE(sc, comp, [net, is_train], 128, 512, scope, down_sample=True, name='block2_0')
    for l in range(1, blocks[1]):
        net = _bottleNeckSE(sc, comp, [net, is_train], 128, 512, scope, name='block2_'+str(l))
    # building block 3
    net = _bottleNeckSE(sc, comp, [net, is_train], 256, 1024, scope, down_sample=True, name='block3_0')
    for l in range(1, blocks[2]):
        net = _bottleNeckSE(sc, comp, [net, is_train], 256, 1024, scope, name='block3_' + str(l))
    # building block 4
    net = _bottleNeckSE(sc, comp, [net, is_train], 512, 2048, scope, down_sample=True, name='block4_0')
    for l in range(1, blocks[3]):
        net = _bottleNeckSE(sc, comp, [net, is_train], 512, 2048, scope, name='block4_' + str(l))
    return sc.assemble(COMP.AVG_POOL(name='avg_pool', inputs=net,
                                     kernel=[7, 7], keep_size=False, if_global=True), sub_scope=scope)

def SE_RESNET_V2_50(sc, comp, inputs, is_train, scope='SE_Res_50'):
    return _se_resnet_v2(sc, comp, inputs, is_train, scope, [3, 4, 6, 3])

def SE_RESNET_V2_101(sc, comp, inputs, is_train, scope='SE_Res_101'):
    return _se_resnet_v2(sc, comp, inputs, is_train, scope, [3, 4, 23, 3])

def SE_RESNET_V2_152(sc, comp, inputs, is_train, scope='SE_Res_152'):
    return _se_resnet_v2(sc, comp, inputs, is_train, scope, [3, 8, 36, 3])