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
from ..kit import ver2num
from ..astro.craft import EMA
from ..power.multiverse import DDP
from ..rule import ENV_RANK

import os
import sys
import torch
from torch.nn import DataParallel as DP
try:
    from ptflops.flops_counter import add_flops_counting_methods
    enable_flops = True
except ImportError:
    enable_flops = False
if not enable_flops:
    try:
        from ptflops.pytorch_engine import add_flops_counting_methods
        enable_flops = True
    except ImportError:
        enable_flops = False

try:
    from graphviz import Digraph
    enable_painting = True
except ImportError:
    print('NEBULAE WARNING ◘ Network graph painting is disabled because of the lack of graphviz library.')
    enable_painting = False


class Inspector(object):

    def __init__(self, export_path='./craft', verbose=True, onnx_ver=-1):
        self.export_path = export_path
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        self.verbose = verbose
        self.onnx_ver = onnx_ver

    def _get_flops(self, archit, *dummy_args, **dummy_kwargs):
        if not enable_flops:
            print('NEBULAE WARNING ◘ ptflops library has not been imported correctly, no operator found.')
            return 0
        flops_model = add_flops_counting_methods(archit)
        flops_model.eval()
        flops_model.start_flops_count(ost=sys.stdout, verbose=False, ignore_list=[])

        _ = flops_model(*dummy_args, **dummy_kwargs)

        flops_count, params_count = flops_model.compute_average_flops_cost()
        flops_model.stop_flops_count()

        return flops_count

    def dissect(self, archit, *dummy_args, **dummy_kwargs):
        rank = int(os.environ.get(ENV_RANK, -1))
        if rank > 0:
            return
        
        if isinstance(archit, (DP, DDP)):
            archit = archit.module
        if isinstance(archit, EMA):
            archit = archit.hull
        if ver2num(torch.__version__) >= ver2num('2.0.0') and isinstance(archit, torch._dynamo.OptimizedModule):
            archit = archit._orig_mod

        archit.eval()
        nbytes = {torch.int8: 1, torch.int64: 8, torch.float16: 2, torch.float32: 4, torch.float64: 8}
        parambytes = sum([p.numel() for p in getattr(archit, 'vars', archit.parameters)()])
        if parambytes<1024:
            parambytes = '%6d  ' % parambytes
        elif parambytes<1048576:
            parambytes = '%6.2f K' % (parambytes / 1024)
        elif parambytes<1073741824:
            parambytes = '%6.2f M' % (parambytes / 1048576)
        else:
            parambytes = '%6.2f B' % (parambytes / 1073741824)

        membytes = sum([p.numel() * nbytes[p.dtype] for p in getattr(archit, 'vars', archit.parameters)()])
        if membytes<1024:
            membytes = '%6d B  ' % membytes
        elif membytes<1048576:
            membytes = '%6.2f KB ' % (membytes / 1024)
        elif membytes<1073741824:
            membytes = '%6.2f MiB' % (membytes / 1048576)
        else:
            membytes = '%6.2f GB ' % (membytes / 1073741824)

        flops = self._get_flops(archit, *dummy_args, **dummy_kwargs)

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
        scope = getattr(archit, 'scope', archit.__class__.__name__)
        print('+' + (len(scope) + 66) * '-' + '+')
        print('| Craft-%s weighs \033[1;32m%s\033[0m with \033[1;32m%s\033[0m FLOPS and \033[1;32m%s\033[0m params |' \
                % (scope, membytes, flops, parambytes))
        print('+' + (len(scope) + 66) * '-' + '+')
        # convert to onnx
        if self.onnx_ver > 0:
            dummies = dummy_args + tuple(dummy_kwargs.values())
            torch.onnx.export(archit, dummies, self.export_path+'.onnx', opset_version=self.onnx_ver)