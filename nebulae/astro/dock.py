#!/usr/bin/env python
# -*- coding:utf-8 -*-
from ..kit.utility import autopad

__all__ = ('coat', 'shell', 'autopad')

from .craft import *
from . import craft
__all__ += craft.__all__


def coat(datum, as_const=True, sync=True):
    raise NotImplementedError('NEBULAE ERROR ៙ coat function becomes valid only after setting up an Engine.')

def shell(datum, as_np=True, sync=False):
    raise NotImplementedError('NEBULAE ERROR ៙ shell function becomes valid only after setting up an Engine.')