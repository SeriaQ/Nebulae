#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import time
import torch

def Timer(joiner=torch.cuda.synchronize):
    def _deco(func):
        def _wrapper(*args, **kwargs):
            t_ = time.time()
            ret = func(*args, **kwargs)
            if callable(joiner):
                joiner()
            _t = time.time()
            if not isinstance(ret, tuple):
                ret = (ret,)
            return (_t - t_,) + ret

        return _wrapper
    return _deco


def SPST(func):
    def _wrapper(*args, **kwargs):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        return func(*args, **kwargs)

    return _wrapper