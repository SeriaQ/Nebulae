#!/usr/bin/env python
'''
component
Created by Seria at 25/11/2018 2:58 PM
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

from functools import partial
import tensorflow as tf

class Pod:
    def __init__(self, comp, symbol, name):
        self.component = comp
        self.symbol = symbol
        self.name = name

    def __rshift__(self, other):
        return Pod([self, other], '>', 'CASCADE')

    def __add__(self, other):
        return Pod([self, other], '+', 'ADD')

    def __sub__(self, other):
        return Pod([self, other], '-', 'SUB')

    def __mul__(self, other):
        return Pod([self, other], '*', 'MUL')

    def __matmul__(self, other):
        return Pod([self, other], '@', 'MATMUL')

    def __and__(self, other):
        return Pod([self, other], '&', 'CONCAT')

    def show(self):
        if isinstance(self.component, list):
            self.component[0].show()
            self.component[1].show()
        else:
            print(self.name)



class Component(object):
    def __init__(self):
        self.warehouse = ['CONV_1D', 'CONV_2D', 'CONV_3D',
                           'SIGMOID', 'TANH', 'SOFTMAX', 'RELU', 'LRELU',
                           'MAX_POOL_2D', 'AVG_POOL_2D']

        self.CONV_1D = 0
        self.CONV = self.conv_2d
        self.CONV_3D = 2
        self.CONV_TRANS = 3
        self.CONV_SEP = 4
        self.CONV_ATROUS = 5

        self.SIGMOID = self.sigmoid
        self.TANH = self.tanh
        self.SOFTMAX = self.softmax
        self.RELU = self.relu
        self.RELU_LEAKY = self.relu_leaky
        self.RELU_EXP = 15

        self.MAX_POOL = self.max_pool_2d
        self.AVG_POOL = self.avg_pool_2d

        self.FLAT = self.flat
        self.DENSE = self.dense

        self.DUPLICATE = self.duplicate

    def addComp(self, name, component):
        if name in self.warehouse:
            raise Exception('%s is an existing component in warehouse.' % name)

    def _createVar(self, name, shape, initializer, regularizer):
        init_err = Exception('%s initializer is not defined or supported.' % initializer)
        # with tf.device('/gpu:0'):
        if initializer == 'xavier':
            var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
        elif initializer == 'trunc_norm':
            var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer())
        elif initializer == 'rand_norm':
            var = tf.get_variable(name, initializer=tf.random_normal(shape, stddev=0.1))
        elif initializer == 'zero':
            var = tf.get_variable(name, shape, initializer=tf.zeros_initializer())
        elif initializer == 'one':
            var = tf.get_variable(name, shape, initializer=tf.ones_initializer())
        else:
            raise init_err

        if regularizer == 'l2':
            weight_decay = tf.nn.l2_loss(var, name=name+'/regularizer')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)
        else:
            raise Exception('%s regularizer is not defined or supported.' % regularizer)

        return var

    def conv_2d(self, **kwargs):
        if 'input' in kwargs.keys():
            symbol = kwargs['input']
        else:
            symbol = ''
        return Pod(partial(self._conv_2d, **kwargs), symbol, kwargs['name'])
    def _conv_2d(self, name, input, kernel_size, in_out_chs, w_init='xavier', b_init=None,
                w_reg='l2', b_reg='l2', stride=(1, 1), padding='same'):
        padding = padding.upper()
        w = self._createVar(name+'_w', kernel_size + in_out_chs, w_init, w_reg)
        if b_init:
            b = self._createVar(name+'_b', [in_out_chs[-1]], b_init, b_reg)
            return tf.nn.bias_add(tf.nn.conv2d(input, w, [1, stride[0], stride[1], 1], padding), b, name=name)
        else:
            return tf.nn.conv2d(input, w, [1, stride[0], stride[1], 1], padding, name=name)

    def sigmoid(self, **kwargs):
        if 'input' in kwargs.keys():
            symbol = kwargs['input']
        else:
            symbol = ''
        return Pod(partial(self._sigmoid, **kwargs), symbol, kwargs['name'])
    def _sigmoid(self, name, input):
        return tf.nn.sigmoid(input, name)

    def tanh(self, **kwargs):
        if 'input' in kwargs.keys():
            symbol = kwargs['input']
        else:
            symbol = ''
        return Pod(partial(self._tanh, **kwargs), symbol, kwargs['name'])
    def _tanh(self, name, input):
        return tf.nn.tanh(input, name)

    def softmax(self, **kwargs):
        if 'input' in kwargs.keys():
            symbol = kwargs['input']
        else:
            symbol = ''
        return Pod(partial(self._tanh, **kwargs), symbol, kwargs['name'])
    def _softmax(self, name, input):
        return tf.nn.softmax(input, name)

    def relu(self, **kwargs):
        if 'input' in kwargs.keys():
            symbol = kwargs['input']
        else:
            symbol = ''
        return Pod(partial(self._relu, **kwargs), symbol, kwargs['name'])
    def _relu(self, name, input):
        return tf.nn.relu(input, name)

    def relu_leaky(self, **kwargs):
        if 'input' in kwargs.keys():
            symbol = kwargs['input']
        else:
            symbol = ''
        return Pod(partial(self._relu_leaky, **kwargs), symbol, kwargs['name'])
    def _relu_leaky(self, name, input):
        return tf.nn.leaky_relu(input, name)

    def max_pool_2d(self, **kwargs):
        if 'input' in kwargs.keys():
            symbol = kwargs['input']
        else:
            symbol = ''
        return Pod(partial(self._max_pool_2d, **kwargs), symbol, kwargs['name'])
    def _max_pool_2d(self, name, input, kernel=(2, 2), stride=(2, 2), padding='same'):
        padding = padding.upper()
        return tf.nn.max_pool(input,
                              [1, kernel[0], kernel[1], 1],
                              [1, stride[0], stride[1], 1],
                              padding, name=name)

    def avg_pool_2d(self, **kwargs):
        if 'input' in kwargs.keys():
            symbol = kwargs['input']
        else:
            symbol = ''
        return Pod(partial(self._avg_pool_2d, **kwargs), symbol, kwargs['name'])
    def _avg_pool_2d(self, name, input, kernel=(2, 2), stride=(2, 2), padding='same'):
        padding = padding.upper()
        return tf.nn.avg_pool(input,
                              [1, kernel[0], kernel[1], 1],
                              [1, stride[0], stride[1], 1],
                              padding, name=name)

    def flat(self, **kwargs):
        if 'input' in kwargs.keys():
            symbol = kwargs['input']
        else:
            symbol = ''
        return Pod(partial(self._flat, **kwargs), symbol, kwargs['name'])
    def _flat(self, name, input):
        return tf.layers.flatten(input, name=name)

    def dense(self, **kwargs):
        if 'input' in kwargs.keys():
            symbol = kwargs['input']
        else:
            symbol = ''
        return Pod(partial(self._dense, **kwargs), symbol, kwargs['name'])
    def _dense(self, name, input, out_chs, w_init='xavier', b_init=None, w_reg='l2', b_reg='l2'):
        in_chs = int(input.get_shape()[-1])
        in_out_chs = [in_chs, out_chs]
        w = self._createVar(name + '_w', in_out_chs, w_init, w_reg)
        if b_init:
            b = self._createVar(name + '_b', [in_out_chs[-1]], b_init, b_reg)
            return tf.nn.bias_add(tf.matmul(input, w), b, name=name)
        else:
            return tf.matmul(input, w, name=name)

    def duplicate(self, **kwargs):
        if 'input' in kwargs.keys():
            symbol = kwargs['input']
        else:
            symbol = ''
        return Pod(partial(self._duplicate, **kwargs), symbol, kwargs['name'])
    def _duplicate(self, name, input):
        return tf.identity(input, name)