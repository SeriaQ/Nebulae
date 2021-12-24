#!/usr/bin/env python
'''
law
Created by Seria at 02/02/2019 12:50 AM
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
class Constant(object):
    CORE = 'PYTORCH'
    FIELD_SEP = '"'
    CHAR_SEP = ','
    FRAME_KEY = '_MAX_FRAMES'
    VALID_DTYPE = ['uint8', 'uint16', 'uint32', 'int8', 'int16', 'int32', 'int64',
                    'float16', 'float32', 'float64', 'str', 'bool']