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

class LayoutSheet(object):

    def __init__(self, layout_sheet_name):
        self.layout_sheet = Digraph(comment='The Space Craft')
        self.ls_name = layout_sheet_name
        self.nodes = []

    def _drawNode(self, prev_node, curr_node, edge, init=False, visible=True):
        if prev_node not in self.nodes:
            self.nodes.append(prev_node)
            if init:
                self.layout_sheet.node(prev_node, prev_node, shape='doublecircle')
        if curr_node not in self.nodes:
            self.layout_sheet.node(curr_node, curr_node, shape='box')
        self.layout_sheet.edge(prev_node, curr_node, label=edge)
        if visible:
            print('| Assembling component\033[34m%30s\033[0m | Output%s |' % (curr_node, edge))
            print('+' + ((60 + len(edge.split('x')) * 7) * '-') + '+')

    def _combineNodes(self, prev_node, prev_shape, curr_node, symbol):
        self.layout_sheet.node(curr_node+symbol, symbol, shape='circle')
        self.nodes.append(curr_node+symbol)
        for p in range(len(prev_node)):
            pn = prev_node[p]
            ps = prev_shape[p]
            if pn not in self.nodes:
                self.nodes.append(prev_node)
                self._drawNode(pn, curr_node+symbol, ps, visible=False)
        self._drawNode(curr_node+symbol, curr_node, '', visible=False)

    def _generateLS(self):
        self.layout_sheet.render(self.ls_name+'.gv', view=False)