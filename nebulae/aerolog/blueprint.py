#!/usr/bin/env python
'''
layout_sheet
Created by Seria at 02/12/2018 1:20 PM
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

from graphviz import Digraph
from ..astrobase.dock import Tensor
import os
import numpy as np

class BluePrint(object):

    def __init__(self, config=None, verbose=True, hidden=[]):
        if config is None:
            self.param = {'verbose': verbose, 'hidden': hidden}
        else:
            self.param = config
        self.layout = Digraph(comment='The Space Craft', format='jpg')
        self.layout.attr('node', shape='doublecircle')
        self.seen = []
        self.displayed = [] # already printed on screen
        self.shapes = {}
        self._await = False
        self._path = './_'

    def paint(self, archit, *slot):
        dummy = []
        core = os.environ.get('NEB_CORE', 'PYTORCH')
        for k,s,t in slot:
            if core == 'TENSORFLOW':
                import tensorflow as tf
                if isinstance(s, tuple):
                    data = np.random.rand(1, *s).astype(t)
                    dummy.append(Tensor(k, tf.covert_to_tensor(data)))
                else:
                    dummy.append(Tensor(k, s))
            # elif core == 'MXNET':
            #     from mxnet import nd
            #     if isinstance(s, tuple):
            #         data = np.random.rand(1, *s).astype(t)
            #         dummy.append(Tensor(k, nd.array(data)))
            #     else:
            #         dummy.append(Tensor(k, s))
            elif core == 'PYTORCH':
                import torch
                if isinstance(s, tuple):
                    data = np.random.rand(1, *s).astype(t)
                    dummy.append(Tensor(k, torch.from_numpy(data)))
                else:
                    dummy.append(Tensor(k, s))
            else:
                raise ValueError('NEBULAE ERROR ⨷ %s is an unsupported core.' % core)
        _ = archit(*tuple(dummy))
        self.archit = archit
        for pod in archit.pods:
            self._parse(pod)

    def _parse(self, pod):
        sym = pod.symbol
        comp = pod.comp
        if sym == '':
            return pod.name
        else:
            nodes = []
            for c in comp:
                node = self._parse(c)
                nodes.append(node)
                if node not in self.seen and node not in self.param['hidden']:
                    self.layout.node(node, node, shape='box')
                    self.seen.append(node)
                    if node not in self.archit.dict.keys():
                        self.shapes[node] = ' '
                    else:
                        self.shapes[node] = ' '+' x '.join([str(s) for s in self.archit[node].shape[1:]])
            if sym == '>':
                for i in range(len(nodes) - 1):
                    prev = nodes[i]
                    curr = nodes[i+1]
                    if prev.symbol=='^':
                        if curr.symbol=='^':
                            if prev not in self.seen:
                                self.layout.node(prev, '^', shape='circle')
                                self.seen.append(prev)
                            pivot = prev
                            for c in curr.comp:
                                self.layout.edge(pivot, c.name, label=self.shapes[pivot])
                        else:
                            assert curr.symbol==''
                            pivot = curr
                        for c in prev.comp:
                            self.layout.edge(c.name, pivot, label=self.shapes[c.name])
                    self.layout.edge(nodes[i], nodes[i + 1], label=self.shapes[nodes[i]])
                    if self.param['verbose']:
                        shape_out = self.shapes[nodes[i+1]]
                        print('| Assembling component\033[34m%32s\033[0m | Output%s |' % (nodes[i+1], shape_out))
                        print('+' + (63 + len(shape_out)) * '-' + '+')
                pod.name = nodes[-1]
                return nodes[-1]
            elif sym in ['+', '-', '*', '@', '&', '|', '^']:
                pivot = sym.join(nodes)
                pod.name = pivot
                if sym=='^':
                    return pivot
                else:
                    if pivot not in self.seen:
                        self.layout.node(pivot, sym, shape='circle')
                        self.seen.append(pivot)
                    for node in nodes:
                        self.layout.edge(node, pivot, label=self.shapes[node])
                    return pivot

    def log(self, log_dir=None, graph_name=None):
        core = os.environ.get('NEB_CORE', 'PYTORCH')
        # TODO: may need to be corrected for TF2.x
        if core=='TENSORFLOW' or self._await:
            if log_dir is None:
                log_dir = os.path.dirname(self._path)
            if graph_name is None:
                graph_name = os.path.basename(self._path)
            gv_name = os.path.join(log_dir, '%s.gv'%graph_name)
            self.layout.render(gv_name, view=False)
            os.rename(gv_name + '.jpg', os.path.join(log_dir, '%s.jpg'%graph_name))
            os.remove(gv_name)