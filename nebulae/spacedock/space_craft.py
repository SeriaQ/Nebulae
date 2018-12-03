#!/usr/bin/env python
'''
space_craft
Created by Seria at 23/11/2018 10:31 AM
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

import tensorflow as tf
import numpy as np

class SpaceCraft(object):
    def __init__(self, scope, reuse=None, layout_sheet=None):
        self.scope = scope
        if reuse is None:
            self.reuse = tf.AUTO_REUSE
        self.valid_dtypes = ['uint8', 'uint16', 'uint32', 'int8', 'int16', 'int32', 'int64',
                             'float16', 'float32', 'float64', 'str', 'bool']
        self.operands = {'+': tf.add, '-': tf.subtract, '*': tf.multiply, '@': tf.matmul, '&': tf.concat}
        self.layout = {}
        if layout_sheet is None:
            self.verbose = False
        else:
            self.layout_sheet = layout_sheet
            self.verbose = True

    def fuelLine(self, name, shape, dtype):
        if dtype not in self.valid_dtypes:
            raise Exception('%s is not a valid data type.' % dtype)
        with tf.variable_scope('FL'):
            self.layout[name] = tf.placeholder(tf.as_dtype(dtype), shape, name)

    def _getHull(self, component):
        shape = component.get_shape()[1:].as_list()
        hull = ' '
        for dim in range(len(shape) - 1):
            hull += '%-4d x ' % shape[dim]
        hull += '%-5d' % shape[-1]
        return hull

    def assembleComp(self, left_comp, right_comp=None, gear='fit', assemblage=None, sub_scope=''):
        if sub_scope:
            scope = self.scope + '/' + sub_scope
            sub_scope += '/'
        else:
            scope = self.scope
        with tf.variable_scope(scope, reuse=self.reuse):
            left_symbol = left_comp.symbol
            if gear == 'fit':
                if not isinstance(left_symbol, str) or not left_symbol: # this is an individual component
                    if assemblage is None:
                        assert not isinstance(left_symbol, str)
                        self.layout[sub_scope+left_comp.name] = left_comp.component()
                        if self.verbose: # draw layout sheet
                            hull = self._getHull(left_symbol)
                            if left_symbol.name.startswith('FL/'): # if the input is a fuel line
                                self.layout_sheet._drawNode(left_symbol.op.name[3:], sub_scope + left_comp.name,
                                                            hull, init=True)
                            else:
                                self.layout_sheet._drawNode(left_symbol.op.name[len(scope)+1:],
                                                            sub_scope + left_comp.name, hull)
                    else:
                        self.layout[sub_scope+left_comp.name] = left_comp.component(input=self.layout[assemblage])
                        if self.verbose:  # draw layout sheet
                            hull = self._getHull(self.layout[assemblage])
                            self.layout_sheet._drawNode(assemblage, sub_scope + left_comp.name, hull)
                    return left_comp.name
                elif left_symbol == '>':
                    for comp in left_comp.component:
                        assemblage = self.assembleComp(comp, assemblage=assemblage, sub_scope=sub_scope)
                    return assemblage
                elif left_symbol in ['+', '-', '*', '@', '&']:
                    operator = None
                    assemblage_name = left_comp.name
                    assemblage_list = []
                    hull_list = []
                    init_assemblage = assemblage
                    for comp in left_comp.component:
                        assemblage = self.assembleComp(comp, assemblage=init_assemblage,
                                                       sub_scope = sub_scope)
                        assemblage_name += '-'+assemblage
                        assemblage_list.append(assemblage)
                        if self.verbose:
                            hull = self._getHull(self.layout[assemblage])
                            hull_list.append(hull)
                        if operator is None:
                            operator = self.layout[assemblage]
                        else:
                            if left_symbol == '&':
                                operator = self.operands[left_symbol]([operator, self.layout[assemblage]],
                                                                       axis=-1, name=assemblage_name)
                            else:
                                operator = self.operands[left_symbol](operator,
                                                                   self.layout[assemblage],
                                                                   name=assemblage_name)
                    self.layout[sub_scope+assemblage_name] = operator
                    if self.verbose:
                        self.layout_sheet._combineNodes(assemblage_list, hull_list,
                                                        sub_scope+assemblage_name, left_symbol)
                    return assemblage_name
                else:
                    raise TypeError('unsupported operand type for %s: "Pod" and "Pod".' % left_comp.symbol)
            # elif gear == ''