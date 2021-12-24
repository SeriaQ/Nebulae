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
import numpy as np
import random as rand
from collections import abc
from torchvision.transforms import *
F = functional
from PIL import Image

from ..toolkit import byte2arr


__all__ = ('Comburant', 'HWC2CHW', 'Random',
           'NEAREST', 'LINEAR', 'CUBIC', 'HORIZONTAL', 'VERTICAL',
           'Resize', 'Crop', 'Flip', 'Rotate',
           'Brighten', 'Contrast', 'Saturate', 'Hue')


NEAREST = 0
LINEAR = 1
CUBIC = 2

PIL_INTERP = {NEAREST: Image.NEAREST, LINEAR: Image.BILINEAR, CUBIC: Image.BICUBIC}

HORIZONTAL = 10
VERTICAL = 11




class Comburant(object):
    def __init__(self, *args, is_encoded=False):
        self.comburant = Compose(list(args))
        self.is_encoded = is_encoded

    def __call__(self, imgs):
        if self.is_encoded:
            if isinstance(imgs, abc.Sequence):
                img = []
                for i in imgs:
                    img.append(byte2arr(i, as_np=False))
            else:
                img = byte2arr(imgs, as_np=False)
            imgs = img
        imgs = self.comburant(imgs)
        if isinstance(imgs, abc.Sequence):
            img = []
            for i in imgs:
                i = np.array(i)
                i = i.astype(np.float32) / 255
                img.append(i)
        else:
            img = np.array(imgs)
            img = img.astype(np.float32) / 255
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
        return np.transpose(img, (2, 0, 1))



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
        # size: (height, width)
        super(Resize, self).__init__()
        self.size = size
        self.interp = interp

    def call(self, img):
        return img.resize((self.size[1], self.size[0]), PIL_INTERP[self.interp])



class Crop(ABC):
    def __init__(self, size, padding=(0, 0, 0, 0), area_ratio=(1, 1), aspect_ratio=(1, 1), interp=LINEAR, scale=()):
        # size: (height, width)
        # padding: (left, top, right, bottom)
        super(Crop, self).__init__()
        self.size = size
        self.padding = padding
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.scale = scale
        if area_ratio == aspect_ratio == (1,1):
            self.reshape = False
            self.comburant = RandomCrop(size)
        else:
            self.reshape = True
            self.comburant = RandomResizedCrop(size, area_ratio, aspect_ratio, PIL_INTERP[interp])

    def call(self, img, param, t=1):
        param = [p * t for p in param]
        y, x, h, w = param
        padding = tuple([p * t for p in self.padding])
        size = tuple([s * t for s in self.size])
        img = F.pad(img, padding, 0, 'constant')
        # pad the width if needed
        if img.size[0] < size[1]:
            img = F.pad(img, (size[1] - img.size[0], 0), 0, 'constant')
        # pad the height if needed
        if img.size[1] < size[0]:
            img = F.pad(img, (0, size[0] - img.size[1]), 0, 'constant')

        if self.reshape:
            return F.resized_crop(img, y, x, h, w, size, self.comburant.interpolation)
        else:
            return F.crop(img, y, x, h, w)


    def __call__(self, imgs):
        if len(self.scale) == 0:
            self.scale = len(imgs) * [1]
        img = F.pad(imgs[0], self.padding, 0, 'constant')
        # pad the width if needed
        if img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), 0, 'constant')
        # pad the height if needed
        if img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), 0, 'constant')
        if self.reshape:
            param = self.comburant.get_params(img, self.comburant.scale, self.comburant.ratio)
        else:
            param = self.comburant.get_params(img, self.comburant.size)

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
        if axial == HORIZONTAL:
            self.comburant = RandomVerticalFlip(1)
        elif axial == VERTICAL:
            self.comburant = RandomHorizontalFlip(1)
        else:
            raise Exception('NEBULAE ERROR ⨷ the invoked flip type is not defined or supported.')

    def call(self, img):
        return self.comburant(img)



class Rotate(ABC):
    def __init__(self, degree, intact=False, interp=NEAREST):
        '''
        Args
        intact: whether to keep image intact which might enlarge the output size
        '''
        super(Rotate, self).__init__()
        self.comburant = RandomRotation(degree, PIL_INTERP[interp], intact)

    def call(self, img, angle):
        return F.rotate(img, angle, self.comburant.resample, self.comburant.expand,
                        self.comburant.center, self.comburant.fill)

    def __call__(self, imgs):
        angle = self.comburant.get_params(self.comburant.degrees)
        if isinstance(imgs, abc.Sequence):
            ret = []
            for i in imgs:
                ret.append(self.call(i, angle))
        else:
            ret = self.call(imgs, angle)

        return ret



class Brighten(ABC):
    def __init__(self, range):
        super(Brighten, self).__init__()
        self.comburant = ColorJitter(brightness=range)

    def call(self, img, factor):
        return F.adjust_brightness(img, factor)

    def __call__(self, imgs):
        factor = rand.uniform(self.comburant.brightness[0], self.comburant.brightness[1])
        if isinstance(imgs, abc.Sequence):
            ret = []
            for i in imgs:
                ret.append(self.call(i, factor))
        else:
            ret = self.call(imgs, factor)

        return ret



class Contrast(ABC):
    def __init__(self, range):
        super(Contrast, self).__init__()
        self.comburant = ColorJitter(contrast=range)

    def call(self, img, factor):
        return F.adjust_contrast(img, factor)

    def __call__(self, imgs):
        factor = rand.uniform(self.comburant.contrast[0], self.comburant.contrast[1])
        if isinstance(imgs, abc.Sequence):
            ret = []
            for i in imgs:
                ret.append(self.call(i, factor))
        else:
            ret = self.call(imgs, factor)

        return ret



class Saturate(ABC):
    def __init__(self, range):
        super(Saturate, self).__init__()
        self.comburant = ColorJitter(saturation=range)

    def call(self, img, factor):
        return F.adjust_saturation(img, factor)

    def __call__(self, imgs):
        factor = rand.uniform(self.comburant.saturation[0], self.comburant.saturation[1])
        if isinstance(imgs, abc.Sequence):
            ret = []
            for i in imgs:
                ret.append(self.call(i, factor))
        else:
            ret = self.call(imgs, factor)

        return ret



class Hue(ABC):
    def __init__(self, range):
        super(Hue, self).__init__()
        self.comburant = ColorJitter(hue=range)

    def call(self, img, factor):
        return F.adjust_hue(img, factor)

    def __call__(self, imgs):
        factor = rand.uniform(self.comburant.hue[0], self.comburant.hue[1])
        if isinstance(imgs, abc.Sequence):
            ret = []
            for i in imgs:
                ret.append(self.call(i, factor))
        else:
            ret = self.call(imgs, factor)

        return ret