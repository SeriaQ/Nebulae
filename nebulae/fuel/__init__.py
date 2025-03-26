#!/usr/bin/env python
# -*- coding:utf-8 -*-
from .tank import *
from .comburant import *
from .generator import *
from . import tank, comburant, generator

from ..kit.utility import _merge_fuel as merge
from ..kit.utility import _fill_fuel as fill
from ..kit.utility import _deduct_fuel as deduct

__all__ = ('merge', 'fill', 'deduct') + comburant.__all__ + generator.__all__ + tank.__all__