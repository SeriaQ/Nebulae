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
from ..astro import fn
from graphviz import Digraph
import os
import torch

class Inspector(object):

    def __init__(self, export_path='./viz', verbose=True, hidden=(), onnx_ver=9):
        self.export_path = export_path
        self.verbose = verbose
        self.hidden = hidden
        self.onnx_ver = onnx_ver
        self.layout = Digraph(comment='The Space Craft', format='jpg')
        self.layout.attr('node', shape='doublecircle')
        self.seen = []
        self.displayed = [] # already printed on screen
        self.shapes = {}

    def paint(self, archit, *dummy_args, **dummy_kwargs):
        rank = int(os.environ.get('RANK', -1))
        if rank > 0:
            return

        _ = archit(*dummy_args, **dummy_kwargs)
        self.archit = archit
        print('+' + 77 * '-' + '+')
        for f in archit.fns:
            self._parse(f)
        self.layout.render(self.export_path, view=False, format='png')
        os.remove(self.export_path)

    def _parse(self, f):
        sym = f.symbol
        comp = f.comp
        if sym == '':
            return f
        else:
            nodes = []
            for c in comp:
                node = self._parse(c)
                nodes.append(node)
                if node.name not in self.seen and node.name not in self.hidden:
                    self.layout.node(node.name, node.name, shape='box')
                    self.seen.append(node.name)
                    if self.archit[node.name] is None:
                        self.shapes[node.name] = ' '
                    else:
                        self.shapes[node.name] = ' '+' x '.join([str(s) for s in self.archit[node.name].shape[1:]])
            if sym == '>':
                for i in range(len(nodes) - 1):
                    prev = nodes[i]
                    curr = nodes[i+1]
                    if prev.symbol=='^':
                        if curr.symbol=='^':
                            if prev not in self.seen:
                                self.layout.node(prev.name, '^', shape='circle')
                                self.seen.append(prev.name)
                            pivot = prev
                            for c in curr.comp:
                                self.layout.edge(pivot.name, c.name, label=self.shapes[pivot.name])
                        else:
                            assert curr.symbol==''
                            pivot = curr
                        for c in prev.comp:
                            self.layout.edge(c.name, pivot.name, label=self.shapes[c.name])
                    self.layout.edge(prev.name, curr.name, label=self.shapes[prev.name])
                    if self.verbose and (prev.name not in self.displayed):
                        shape_out = self.shapes[prev.name]
                        print('| Component\033[34m%32s\033[0m | Output%25s |' % (prev.name, shape_out))
                        print('+' + 77 * '-' + '+')
                        self.displayed.append(prev.name)
                if self.verbose and (curr.name not in self.displayed):
                    shape_out = self.shapes[curr.name]
                    print('| Component\033[34m%32s\033[0m | Output%25s |' % (curr.name, shape_out))
                    print('+' + 77 * '-' + '+')
                    self.displayed.append(curr.name)
                f.name = nodes[-1].name
                return nodes[-1]
            elif sym in ['+', '-', '*', '@', '&', '|', '^']:
                pivot = fn(sym.join([nd.name for nd in nodes]))
                f.name = pivot.name
                self.shapes[pivot.name] = ' '
                if sym=='^':
                    return pivot
                else:
                    if pivot.name not in self.seen:
                        self.layout.node(pivot.name, sym, shape='circle')
                        self.seen.append(pivot.name)
                    if pivot.name not in self.displayed:
                        self.displayed.append(pivot.name)
                    for node in nodes:
                        self.layout.edge(node.name, pivot.name, label=self.shapes[node.name])
                    return pivot

    def dissect(self, *dummy_args, **dummy_kwargs):
        rank = int(os.environ.get('RANK', -1))
        if rank > 0:
            return
        # TODO: draw architectures

        nbytes = {torch.int8: 1, torch.int64: 8, torch.float16: 2, torch.float32: 4, torch.float64: 8}
        parambytes = sum([p.numel() * nbytes[p.dtype] for p in self.vars()])
        if parambytes<1024:
            parambytes = '%6d B  ' % parambytes
        elif parambytes<1048576:
            parambytes = '%6.2f KB ' % (parambytes / 1024)
        elif parambytes<1073741824:
            parambytes = '%6.2f MiB' % (parambytes / 1048576)
        else:
            parambytes = '%6.2f GB ' % (parambytes / 1073741824)

        flops = self._get_flops(*dummy_args, **dummy_kwargs)

        if flops<1024:
            flops = '%6d  ' % flops
        elif flops<1048576:
            flops = '%6.2f K' % (flops / 1024)
        elif flops<1073741824:
            flops = '%6.2f M' % (flops / 1048576)
        elif flops<1099511627776:
            flops = '%6.2f G' % (flops / 1073741824)
        else:
            flops = '%6.2f T' % (flops / 1099511627776)
        print('+' + (len(self.scope) + 46) * '-' + '+')
        print('| Craft-%s weighs \033[1;32m%s\033[0m with \033[1;32m%s\033[0m FLOPS |' % (self.scope, parambytes, flops))
        print('+' + (len(self.scope) + 46) * '-' + '+')

        # convert to onnx
        if self.verbose:
            dummies = dummy_args + tuple(dummy_kwargs.values())
            torch.onnx.export(self, dummies, self.export_path, opset_version=self.onnx_ver)