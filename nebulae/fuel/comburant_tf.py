#!/usr/bin/env python
'''
Created by Seria at 02/11/2018 3:38 PM
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
import math
import numpy as np
import random as rand
from collections import abc
from tensorflow import image as tfimg
import tensorflow as tf
import scipy.ndimage as ndimg


__all__ = ('Comburant', 'HWC2CHW', 'Random',
           'NEAREST', 'LINEAR', 'CUBIC', 'HORIZONTAL', 'VERTICAL',
           'Resize', 'Crop', 'Flip', 'Rotate',
           'Brighten', 'Contrast', 'Saturate', 'Hue')


NEAREST = 0
LINEAR = 1
CUBIC = 2

TF_INTERP = {NEAREST: 'nearest', LINEAR: 'bilinear', CUBIC: 'bicubic'}
SP_INTERP = {NEAREST: 0, LINEAR: 1, CUBIC: 3}

HORIZONTAL = 10
VERTICAL = 11




class Comburant(object):
    def __init__(self, *args, is_encoded=False):
        self.comburant = list(args)
        self.is_encoded = is_encoded

    def __call__(self, img):
        if self.is_encoded:
            img = tfimg.decode_jpeg(img)
            img = tf.image.convert_image_dtype(img, tf.float32)
        for cbr in self.comburant:
            img = cbr(img)
        if not tf.is_tensor(img):
            img = tf.convert_to_tensor(img)
        return img



class ABC(object):
    def __init__(self):
        pass

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, imgs):
        if isinstance(imgs, abc.Sequence):
            ret = []
            for i in imgs:
                ret.append(self.call(i))
        else:
            ret = self.call(imgs)

        return ret



class HWC2CHW(ABC):
    def __init__(self):
        super(HWC2CHW, self).__init__()

    def call(self, img):
        return tf.transpose(img, (2, 0, 1))



class Random(ABC):
    def __init__(self, p, comburant):
        super(Random, self).__init__()
        self.p = p
        self.cbr = comburant

    def call(self, img, conduct):
        if conduct:
            return self.cbr(img)
        else:
            return img

    def __call__(self, imgs):
        if rand.random() < self.p:
            conduct = True
        else:
            conduct = False
        if isinstance(imgs, abc.Sequence):
            ret = []
            for i in imgs:
                ret.append(self.call(i, conduct))
        else:
            ret = self.call(imgs, conduct)

        return ret



class Resize(ABC):
    def __init__(self, size, interp=LINEAR):
        super(Resize, self).__init__()
        # size: (height, width)
        self.size = size
        self.interp = interp

    def call(self, img):
        return tfimg.resize(img, self.size, TF_INTERP[self.interp])



class Crop(ABC):
    def __init__(self, size: tuple, padding=(0, 0, 0, 0), area_ratio=(1, 1), aspect_ratio=(1, 1),
                 interp=LINEAR, scale=()):
        # size: (height, width)
        # padding: (left, top, right, bottom)
        super(Crop, self).__init__()
        self.size = size
        self.scale = scale
        self.padding = padding
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.interp = interp
        if self.area_ratio == self.aspect_ratio == (1, 1):
            self.reshape = False
        else:
            self.reshape = True

    def call(self, img, param, t=1):
        param = [p * t for p in param]
        h, w, c = img.shape
        padding = tuple([p * t for p in self.padding])
        size = tuple([s * t for s in self.size])
        l, t, r, b = padding
        _h = max(h + t + b, size[0])
        _w = max(w + l + r, size[1])
        img = tfimg.pad_to_bounding_box(img, t, l, _h, _w)

        if self.area_ratio == self.aspect_ratio == (1, 1):
            x, y = param
            img = img[y:y+size[0], x:x+size[1]]
        else:
            x, y, cols, rows = param
            img = tfimg.resize(img[y:y+rows, x:x+cols], size, TF_INTERP[self.interp])
        return img

    def __call__(self, imgs):
        if len(self.scale) == 0:
            self.scale = len(imgs) * [1]

        h, w, c = imgs[0].shape
        l, t, r, b = self.padding
        h = max(h + t + b, self.size[0])
        w = max(w + l + r, self.size[1])
        if self.reshape:
            sqrt_aspect_ratio = math.sqrt(rand.random()
                                          * (self.aspect_ratio[1] - self.aspect_ratio[0]) + self.aspect_ratio[0])
            sqrt_area_ratio = math.sqrt(rand.random()
                                        * (self.area_ratio[1] - self.area_ratio[0]) + self.area_ratio[0])
            cols = int(sqrt_area_ratio * w * sqrt_aspect_ratio)
            rows = int(sqrt_area_ratio * h / sqrt_aspect_ratio)
            x = int(rand.random() * (w - cols + 1))
            y = int(rand.random() * (h - rows + 1))
            param = x, y, cols, rows
        else:
            x = int(rand.random() * (w - self.size[1] + 1))
            y = int(rand.random() * (h - self.size[0] + 1))
            param = x, y

        if isinstance(imgs, abc.Sequence):
            ret = []
            for i, v in enumerate(imgs):
                ret.append(self.call(v, param, self.scale[i]))
        else:
            ret = self.call(imgs, param)

        return ret



class Flip(ABC):
    def __init__(self, axial):
        super(Flip, self).__init__()
        self.axial = axial

    def call(self, img):
        if self.axial == HORIZONTAL:
            img = tfimg.flip_up_down(img)
        elif self.axial == VERTICAL:
            img = tfimg.flip_left_right(img)
        else:
            raise Exception('NEBULAE ERROR ⨷ the invoked flip type is not defined or supported.')
        return img



class Rotate(ABC):
    def __init__(self, degree, intact=False, interp=NEAREST):
        '''
        Args
        intact: whether to keep image intact which might enlarge the output size
        '''
        super(Rotate, self).__init__()
        self.degree = degree
        self.intact = intact
        self.interp = interp

    def rot(self, img):
        return ndimg.rotate(img, self.angle, reshape=self.intact, order=SP_INTERP[self.interp])

    # def call(self, img):
    #     h, w, c = img.shape
    #     rot_center = (w / 2., h / 2.)
    #     degree = self.degree * (rand.random() * 2 - 1)
    #     angle = math.radians(degree)
    #     matrix = [
    #         round(math.cos(angle), 15), round(math.sin(angle), 15), 0.,
    #         round(-math.sin(angle), 15), round(math.cos(angle), 15), 0.,
    #     ]
    #
    #     def transform(x, y, matrix):
    #         (a, b, c, d, e, f) = matrix
    #         return a * x + b * y + c, d * x + e * y + f
    #
    #     matrix[2], matrix[5] = transform(-rot_center[0], -rot_center[1], matrix)
    #     matrix[2] += rot_center[0]
    #     matrix[5] += rot_center[1]
    #
    #     if self.intact:
    #         # calculate output size
    #         xx = []
    #         yy = []
    #         for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
    #             x, y = transform(x, y, matrix)
    #             xx.append(x)
    #             yy.append(y)
    #         nw = math.ceil(max(xx)) - math.floor(min(xx))
    #         nh = math.ceil(max(yy)) - math.floor(min(yy))
    #
    #         # We multiply a translation matrix from the right.  Because of its
    #         # special form, this is the same as taking the image of the
    #         # translation vector as new translation vector.
    #         matrix[2], matrix[5] = transform(-(nw - w) / 2.0, -(nh - h) / 2.0, matrix)
    #     else:
    #         nw, nh = w, h
    #
    #     rot_img = np.zeros((nh, nw, c), dtype=img.dtype)
    #     zeros = np.zeros((c,), dtype=img.dtype)
    #     for i in range(nh):
    #         for j in range(nw):
    #             x, y = transform(j, i, matrix)
    #             x, y = round(x), round(y)
    #             if x >= w or y >= h or x < 0 or y < 0:
    #                 pix = zeros
    #             else:
    #                 pix = img[round(y), round(x)]
    #             rot_img[i, j] = pix
    #
    #     return rot_img

    def call(self, img):
        shape = img.shape
        img = tf.numpy_function(self.rot, [img], tf.float32)
        img.set_shape(shape)
        return img

    def __call__(self, imgs):
        self.angle = self.degree * (rand.random() * 2 - 1)
        if isinstance(imgs, abc.Sequence):
            ret = []
            for i in imgs:
                ret.append(self.call(i))
        else:
            ret = self.call(imgs)

        return ret



class Brighten(ABC):
    def __init__(self, range):
        super(Brighten, self).__init__()
        self.range = range

    def call(self, img): ############################################## TODO: add __call__
        return tfimg.random_brightness(img, self.range)



class Contrast(ABC):
    def __init__(self, range):
        super(Contrast, self).__init__()
        self.range = range

    def call(self, img): ############################################## TODO: add __call__
        return tfimg.random_contrast(img, max(0, 1-self.range), 1+self.range)



class Saturate(ABC):
    def __init__(self, range):
        super(Saturate, self).__init__()
        self.range = range

    def call(self, img): ############################################## TODO: add __call__
        return tfimg.random_saturation(img, max(0, 1-self.range), 1+self.range)



class Hue(ABC):
    def __init__(self, range):
        super(Hue, self).__init__()
        self.range = range

    def call(self, img): ############################################## TODO: add __call__
        return tfimg.random_hue(img, self)