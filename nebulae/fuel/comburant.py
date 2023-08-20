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
from scipy import special, signal
import random as rand
import logging
import math
import cv2
from PIL import Image, ImageFilter
from torchvision.transforms import *
F = functional
from collections import abc
from io import BytesIO
from ..kit import byte2arr
try:
    import av
    logging.getLogger('libav').setLevel(logging.FATAL)
except:
    print('NEBULAE WARNING ◘ that PyAV has not been installed properly might result in failure of video augmentation.')


__all__ = ('Comburant',
           'HW2CHW', 'HWC2CHW', 'Whiten', 'Random',
           'multiple', 'End', 'Void',
           'NEAREST', 'LINEAR', 'CUBIC', 'HORIZONTAL', 'VERTICAL',
           'Pad', 'Resize',
           'Crop', 'Flip', 'Rotate',
           'Brighten', 'Contrast', 'Saturate', 'Hue', 'Blur', 'Sharpen',
           'Noise', 'Sinc',
           'JPEG', 'WebP', 'MPEG4', 'H264', 'VP9', 'AV1')


NUMPY = 0
PIL = 1

NEAREST = 10
LINEAR = 11
CUBIC = 12

CV_INTER = {NEAREST: cv2.INTER_NEAREST, LINEAR: cv2.INTER_LINEAR, CUBIC: cv2.INTER_CUBIC}
PIL_INTERP = {NEAREST: Image.NEAREST, LINEAR: Image.BILINEAR, CUBIC: Image.BICUBIC}

HORIZONTAL = 20
VERTICAL = 21

GAUSSIAN = 30
POISSON = 31




class Comburant(object):
    def __init__(self, *args, format=NUMPY, is_encoded=False):
        if format == NUMPY:
            pass
        elif format == PIL:
            pass
        else:
            raise Exception('NEBULAE ERROR ៙ the image format is not defined or supported.')
        self.format = format
        ls_args = list(args)
        for a in args:
            a._format = format
        self.comburants = ls_args
        self.is_encoded = is_encoded

    def _byte2arr(self, imgs):
        if isinstance(imgs, abc.Sequence):
            arr = []
            for i in imgs:
                arr.append(self._byte2arr(i))
        else:
            if self.is_encoded:
                arr = byte2arr(imgs, as_np=self.format==NUMPY)
            elif imgs.dtype != np.uint8:
                arr = (imgs * 255).astype(np.uint8)
            else:
                arr = imgs
        return arr

    def _post(self, src):
        if isinstance(src, abc.Sequence):
            dst = []
            for img in src:
                dst.append(self._post(img))
        else:
            if not isinstance(src, np.ndarray):
                src = np.array(src)
                dst = src.astype(np.float32) / 255.
            else:
                dst = src.astype(np.float32) / 255.
        return dst

    def __call__(self, src):
        # >| is_encoded = True:  decode image bytes
        # >| is_encoded = False: convert to uint8 for numpy format
        if self.is_encoded or self.format == NUMPY:
            src = self._byte2arr(src)
        # >| go through comburants
        for cbr in self.comburants:
            if isinstance(cbr, End):
                src = self._post(src)
            else:
                src = cbr(src)
        return src



class ABC(object):
    def __init__(self, pair_fn=None):
        self.pair_fn = pair_fn
        self._format = None

    def call(self, *args):
        raise NotImplementedError

    def exec(self, imgs, *args):
        if isinstance(imgs, abc.Sequence):
            ret = []
            if isinstance(imgs[0], abc.Sequence):
                basket = []
                for v in zip(*imgs):
                    basket.append(self.exec(v, *args))
                for r in zip(*basket):
                    ret.append(list(r))
            else:
                if self.pair_fn is None:
                    for i in imgs:
                        ret.append(self.call(i, *args))
                else:
                    params = self.pair_fn(args)
                    for idx, i in enumerate(imgs):
                        ret.append(self.call(i, *(params[idx])))
        else:
            ret = self.call(imgs, *args)

        return ret



class HW2CHW(ABC):
    def __init__(self):
        super(HW2CHW, self).__init__()

    def call(self, img):
        if self._format == NUMPY:
            return np.expand_dims(img, 0)
        elif self._format == PIL:
            raise Exception('NEBULAE ERROR ៙ HW2CHW is not for PIL images.')

    def __call__(self, imgs):
        ret = self.exec(imgs)
        return ret



class HWC2CHW(ABC):
    def __init__(self):
        super(HWC2CHW, self).__init__()

    def call(self, img):
        if self._format == NUMPY:
            return np.transpose(img, (2, 0, 1))
        elif self._format == PIL:
            raise Exception('NEBULAE ERROR ៙ HW2CHW is not for PIL images.')

    def __call__(self, imgs):
        ret = self.exec(imgs)
        return ret



class Whiten(ABC):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(Whiten, self).__init__()
        if isinstance(mean, (tuple, list)):
            self.mean = np.array(mean, dtype=np.float32)[np.newaxis, :, np.newaxis, np.newaxis]
        elif isinstance(mean, (float, int)):
            self.mean = float(mean)
        else:
            raise Exception('NEBULAE ERROR ៙ mean value must be a number or array.')
        if isinstance(std, (tuple, list)):
            self.std = np.array(std, dtype=np.float32)[np.newaxis, :, np.newaxis, np.newaxis]
        elif isinstance(std, (float, int)):
            self.std = float(std)
        else:
            raise Exception('NEBULAE ERROR ៙ std value must be a number or array.')

    def call(self, img):
        if self._format == NUMPY:
            return (img - self.mean) / self.std
        elif self._format == PIL:
            raise Exception('NEBULAE ERROR ៙ HW2CHW is not for PIL images.')

    def __call__(self, imgs):
        ret = self.exec(imgs)
        return ret



class Random(object):
    def __init__(self, p, comburant):
        super(Random, self).__init__()
        if isinstance(p, (tuple, list)):
            assert len(p)==len(comburant), 'NEBULAE ERROR ៙ the number of prob does not match with comburants.'
            assert sum(p)==1, 'NEBULAE ERROR ៙ the sum of probabilities must be 1.'
        self.p = p
        self.cbr = comburant

    def __call__(self, imgs):
        if not isinstance(self.p, (tuple, list)):
            if rand.random() < self.p:
                self.cbr._format = self._format
                return self.cbr(imgs)
            else:
                return imgs
        else:
            cbr = np.random.choice(self.cbr, size=1, replace=False, p=self.p)[0]
            cbr._format = self._format
            return cbr(imgs)



def multiple(scales):
    if isinstance(scales, abc.Sequence):
        def _multiply(x, s):
            if isinstance(x, abc.Sequence):
                ret = []
                for v in x:
                    ret.append(_multiply(v, s))
                return ret
            else:
                return x * s
        return lambda x: [_multiply(x, s) for s in scales]
    else:
        raise TypeError('NEBULAE ERROR ៙ non-iterable scales does not make sense.')



class End(object):
    def __init__(self):
        pass

    def __call__(self):
        pass



class Void():
    def __init__(self):
        pass

    def __call__(self, imgs):
        return imgs



class Pad(ABC):
    def __init__(self, padding, pair_fn=None):
        # padding: (top, bottom, left, right)
        super(Pad, self).__init__(pair_fn)
        self.padding = padding

    def _pad_np(self, img, padding):
        if img.ndim == 2:
            img = np.pad(img, ((padding[0], padding[1]), (padding[2], padding[3])), constant_values=0.)
        elif img.ndim == 3:
            img = np.pad(img, ((padding[0], padding[1]), (padding[2], padding[3]), (0, 0)), constant_values=0.)
        else:
            raise AssertionError('NEBULAE ERROR ៙ the dimension of an image is either 2 or 3.')
        return img

    def _pad_pil(self, img, padding):
        padding = (padding[2], padding[0], padding[3], padding[1])
        return F.pad(img, padding, 0, 'constant')

    def call(self, img, padding):
        if self._format == NUMPY:
            return self._pad_np(img, padding)
        elif self._format == PIL:
            return self._pad_pil(img, padding)

    def __call__(self, imgs):
        ret = self.exec(imgs, self.padding)
        return ret



class Resize(ABC):
    def __init__(self, size, interp=LINEAR, pair_fn=None):
        # size: (height, width)
        super(Resize, self).__init__(pair_fn)
        self.size = size
        self.interp = interp

    def _resize_np(self, img, w, h):
        return cv2.resize(img, (w, h), interpolation=CV_INTER[self.interp])

    def _resize_pil(self, img, w, h):
        return img.resize((w, h), PIL_INTERP[self.interp])

    def call(self, img, w, h):
        if self._format == NUMPY:
            return self._resize_np(img, w, h)
        elif self._format == PIL:
            return self._resize_pil(img, w, h)

    def __call__(self, imgs):
        ret = self.exec(imgs, self.size[1], self.size[0])
        return ret



class Crop(ABC):
    def __init__(self, size, area_ratio=(1, 1), aspect_ratio=(1, 1), central=False, interp=LINEAR, pair_fn=None):
        # size: (height, width)
        super(Crop, self).__init__(pair_fn)
        self.size = size
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.central = central
        self.interp = interp
        if area_ratio == aspect_ratio == (1,1):
            self._reshape = False
        else: # firstly crop a patch that meets area and aspect ratio, then resize it to the given size
            self._reshape = True

    def _get_param_np(self, ref):
        height, width = ref.shape[:2]
        height = max(height, self.size[0])
        width = max(width, self.size[1])
        if self._reshape:
            sqrt_aspect_ratio = math.sqrt(rand.random() *
                                          (self.aspect_ratio[1] - self.aspect_ratio[0]) + self.aspect_ratio[0])
            sqrt_area_ratio = math.sqrt(rand.random() *
                                        (self.area_ratio[1] - self.area_ratio[0]) + self.area_ratio[0])
            for _ in range(10):
                w = int(sqrt_area_ratio * width * sqrt_aspect_ratio)
                h = int(sqrt_area_ratio * height / sqrt_aspect_ratio)
                if 0 < w <= width and 0 < h <= height:
                    if self.central:
                        x = (width - w) // 2
                        y = (height - h) // 2
                    else:
                        y = rand.randint(0, height - h)
                        x = rand.randint(0, width - w)
                    return y, x, h, w
            # Fallback to central crop
            in_ratio = float(width) / float(height)
            if (in_ratio < self.aspect_ratio[0]):
                w = width
                h = int(round(w / self.aspect_ratio[0]))
            elif (in_ratio > self.aspect_ratio[1]):
                h = height
                w = int(round(h * self.aspect_ratio[1]))
            else: # whole image
                w = width
                h = height
            y = (height - h) // 2
            x = (width - w) // 2
            return y, x, h, w
        else:
            w = self.size[1]
            h = self.size[0]
            if self.central:
                x = (width - w) // 2
                y = (height - h) // 2
            else:
                x = int(rand.random() * (width - w + 1))
                y = int(rand.random() * (height - h + 1))
            return y, x, h, w

    def _crop_np(self, img, y, x, h, w, size):
        height, width = img.shape[:2]
        if img.ndim == 2:
            # pad the width if needed
            if width < w:
                half = (w - width) // 2
                img = np.pad(img, ((0, 0), (half, w - width - half)), constant_values=0.)
            # pad the height if needed
            if height < h:
                half = (h - height) // 2
                img = np.pad(img, ((half, h - height - half), (0, 0)), constant_values=0.)
        elif img.ndim == 3:
            # pad the width if needed
            if width < w:
                half = (w - width) // 2
                img = np.pad(img, ((0, 0), (half, w - width - half), (0, 0)), constant_values=0.)
            # pad the height if needed
            if height < h:
                half = (h - height) // 2
                img = np.pad(img, ((half, h - height - half), (0, 0), (0, 0)), constant_values=0.)
        else:
            raise AssertionError('NEBULAE ERROR ៙ the dimension of an image is either 2 or 3.')

        if self._reshape:
            cropped = cv2.resize(img[y:y + h, x:x + w], (size[1], size[0]), interpolation=CV_INTER[self.interp])
        else:
            cropped = img[y:y + h, x:x + w]
        return cropped

    def _get_param_pil(self, ref):
        # pad the width if needed
        if ref.size[0] < self.size[1]:
            ref = F.pad(ref, (self.size[1] - ref.size[0], 0), 0, 'constant')
        # pad the height if needed
        if ref.size[1] < self.size[0]:
            ref = F.pad(ref, (0, self.size[0] - ref.size[1]), 0, 'constant')
        if self._reshape:
            self.comburant = RandomResizedCrop(self.size, self.area_ratio, self.aspect_ratio, PIL_INTERP[self.interp])
            param = self.comburant.get_params(ref, self.comburant.scale, self.comburant.ratio)
        else:
            self.comburant = RandomCrop(self.size)
            param = self.comburant.get_params(ref, self.comburant.size)
        return param

    def _crop_pil(self, img, y, x, h, w, size):
        # pad the width if needed
        if img.size[0] < size[1]:
            img = F.pad(img, (size[1] - img.size[0], 0), 0, 'constant')
        # pad the height if needed
        if img.size[1] < size[0]:
            img = F.pad(img, (0, size[0] - img.size[1]), 0, 'constant')

        if self._reshape:
            return F.resized_crop(img, y, x, h, w, size, self.comburant.interpolation)
        else:
            if self.central:
                return F.center_crop(img, size)
            else:
                return F.crop(img, y, x, h, w)

    def call(self, img, y, x, h, w, size):
        if self._format == NUMPY:
            return self._crop_np(img, y, x, h, w, size)
        elif self._format == PIL:
            return self._crop_pil(img, y, x, h, w, size)

    def __call__(self, imgs):
        if self._format == NUMPY:
            img = imgs
            while isinstance(img, abc.Sequence):
                img = img[0]
            y, x, h, w = self._get_param_np(img)
        else:
            img = imgs
            while isinstance(img, abc.Sequence):
                img = img[0]
            y, x, h, w = self._get_param_pil(img) # it has no effect if central is True

        ret = self.exec(imgs, y, x, h, w, self.size)
        return ret



class Flip(ABC):
    def __init__(self, axial):
        super(Flip, self).__init__()
        self.axial = axial

    def _flip_np(self, img):
        if self.axial == HORIZONTAL:
            dst = np.flip(img, 0)
        elif self.axial == VERTICAL:
            dst = np.flip(img, 1)
        else:
            raise Exception('NEBULAE ERROR ៙ the invoked flip type is not defined or supported.')
        return dst

    def _flip_pil(self, img):
        if self.axial == HORIZONTAL:
            comburant = RandomVerticalFlip(1)
        elif self.axial == VERTICAL:
            comburant = RandomHorizontalFlip(1)
        else:
            raise Exception('NEBULAE ERROR ៙ the invoked flip type is not defined or supported.')
        return comburant(img)

    def call(self, img):
        if self._format == NUMPY:
            return self._flip_np(img)
        elif self._format == PIL:
            return self._flip_pil(img)

    def __call__(self, imgs):
        ret = self.exec(imgs)
        return ret



class Rotate(ABC):
    def __init__(self, degree, intact=False, interp=NEAREST):
        '''
        Args
        intact: whether to keep image intact which might enlarge the output size
        '''
        super(Rotate, self).__init__()
        if isinstance(degree, (tuple, list)):
            assert degree[1] >= degree[0], 'NEBULAE ERROR ៙ the second element should not be less than the first.'
        elif isinstance(degree, (int, float)):
            assert abs(degree) <= 360, 'NEBULAE ERROR ៙ a valid degree should have its absolute value under 360.'
        else:
            raise TypeError('NEBULAE ERROR ៙ a valid factor should be an float or a tuple.')
        self.degree = degree
        self.intact = intact
        self.interp = interp

    def _rotate_np(self, img, degree):
        h, w = img.shape[:2]
        mtx = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)
        if self.intact:
            rot_vtx = []
            vertices = [np.array([0., 0.]), np.array([0., h]), np.array([w, 0.]), np.array([float(w), float(h)])]
            for v in vertices:
                rot_vtx.append(mtx[:, :2] @ v.T + mtx[:, 2])
            rot_vtx = np.stack(rot_vtx, axis=1)
            w_prime = math.ceil(rot_vtx[0].max() - rot_vtx[0].min())
            h_prime = math.ceil(rot_vtx[1].max() - rot_vtx[1].min())
            mtx[0, -1] += (w_prime - w) / 2
            mtx[1, -1] += (h_prime - h) / 2
            dst = cv2.warpAffine(img, mtx, (w_prime, h_prime))
        else:
            dst = cv2.warpAffine(img, mtx, (w, h))
        return dst

    def _rotate_pil(self, img, degree):
        return F.rotate(img, degree, self.comburant.resample, self.comburant.expand,
                        self.comburant.center, self.comburant.fill)

    def call(self, img, degree):
        if self._format == NUMPY:
            return self._rotate_np(img, degree)
        elif self._format == PIL:
            return self._rotate_pil(img, degree)

    def __call__(self, imgs):
        if self._format == NUMPY:
            if isinstance(self.degree, (tuple, list)):
                degree = rand.random() * (self.degree[1] - self.degree[0]) + self.degree[0]
            else:
                degree = self.degree
        else:
            self.comburant = RandomRotation(self.degree, PIL_INTERP[self.interp], self.intact)
            if isinstance(self.degree, (tuple, list)):
                degree = self.comburant.get_params(self.comburant.degrees)
            else:
                degree = self.degree
        ret = self.exec(imgs, degree)
        return ret



class Brighten(ABC):
    def __init__(self, factor):
        super(Brighten, self).__init__()
        if isinstance(factor, (tuple, list)):
            assert factor[1] >= factor[0] and factor[0] >= 0, \
                'NEBULAE ERROR ៙ the second element should not be less than the first one and they are both non-negative.'
        elif isinstance(factor, (int, float)):
            assert factor >= 0, 'NEBULAE ERROR ៙ a valid factor should be non-negative.'
        else:
            raise TypeError('NEBULAE ERROR ៙ a valid factor should be an float or a tuple.')
        self.factor = factor

    def _brighten_np(self, img, factor):
        dst = np.round(img * float(factor)) # pixels range between [theta, 1+theta]
        dst = np.clip(dst, 0, 255).astype(np.uint8) # pixels range between [0, 1]
        return dst

    def _brighten_pil(self, img, factor):
        return F.adjust_brightness(img, factor)

    def call(self, img, factor):
        if self._format == NUMPY:
            return self._brighten_np(img, factor)
        elif self._format == PIL:
            return self._brighten_pil(img, factor)

    def __call__(self, imgs):
        if isinstance(self.factor, (tuple, list)):
            factor = rand.uniform(self.factor[0], self.factor[1])
        else:
            factor = self.factor
        ret = self.exec(imgs, factor)
        return ret



class Contrast(ABC):
    def __init__(self, factor):
        super(Contrast, self).__init__()
        if isinstance(factor, (tuple, list)):
            assert factor[1] >= factor[0] and factor[0] >= 0, \
                'NEBULAE ERROR ៙ the second element should not be less than the first one and they are both non-negative.'
        elif isinstance(factor, (int, float)):
            assert factor >= 0, 'NEBULAE ERROR ៙ a valid factor should be non-negative.'
        else:
            raise TypeError('NEBULAE ERROR ៙ a valid factor should be an float or a tuple.')
        self.factor = factor

    def _contrast_np(self, img, factor):
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        white = np.ones_like(img)
        dst = cv2.addWeighted(img, factor, white, gray.mean() * (1. - factor), 0)
        return dst

    def _contrast_pil(self, img, factor):
        return F.adjust_contrast(img, factor)

    def call(self, img, factor):
        if self._format == NUMPY:
            return self._contrast_np(img, factor)
        elif self._format == PIL:
            return self._contrast_pil(img, factor)

    def __call__(self, imgs):
        if isinstance(self.factor, (tuple, list)):
            factor = rand.uniform(self.factor[0], self.factor[1])
        else:
            factor = self.factor
        ret = self.exec(imgs, factor)
        return ret



class Saturate(ABC):
    def __init__(self, factor):
        super(Saturate, self).__init__()
        if isinstance(factor, (tuple, list)):
            assert factor[1] >= factor[0] and factor[0] >= 0, \
                'NEBULAE ERROR ៙ the second element should not be less than the first one and they are both non-negative.'
        elif isinstance(factor, (int, float)):
            assert factor >= 0, 'NEBULAE ERROR ៙ a valid factor should be non-negative.'
        else:
            raise TypeError('NEBULAE ERROR ៙ a valid factor should be an float or a tuple.')
        self.factor = factor

    def _saturate_np(self, img, factor):
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        dst = cv2.addWeighted(img, factor, gray, 1. - factor, 0)
        return dst

    def _saturate_pil(self, img, factor):
        return F.adjust_saturation(img, factor)

    def call(self, img, factor):
        if self._format == NUMPY:
            return self._saturate_np(img, factor)
        elif self._format == PIL:
            return self._saturate_pil(img, factor)

    def __call__(self, imgs):
        if isinstance(self.factor, (tuple, list)):
            factor = rand.uniform(self.factor[0], self.factor[1])
        else:
            factor = self.factor
        ret = self.exec(imgs, factor)
        return ret



class Hue(ABC):
    def __init__(self, factor):
        # hue range: [-0.5, 0.5]
        super(Hue, self).__init__()
        if isinstance(factor, (tuple, list)):
            assert factor[1] >= factor[0] and factor[0] >= -0.5 and factor[1] <= 0.5, \
                'NEBULAE ERROR ៙ the second element should not be less than the first one' \
                + ' and their absolute values are less than 0.5.'
        elif isinstance(factor, (int, float)):
            assert abs(factor) <= 0.5, 'NEBULAE ERROR ៙ a valid degree should have its absolute value under 0.5.'
        else:
            raise TypeError('NEBULAE ERROR ៙ a valid factor should be an float or a tuple.')
        self.factor = factor

    def _hue_np(self, img, factor):
        if img.ndim == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
            hsv[:, :, 0] += (factor * 255 * np.ones(hsv.shape[:2])).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        else:
            return img

    def _hue_pil(self, img, factor):
        return F.adjust_hue(img, factor)

    def call(self, img, factor):
        if self._format == NUMPY:
            return self._hue_np(img, factor)
        elif self._format == PIL:
            return self._hue_pil(img, factor)

    def __call__(self, imgs):
        if isinstance(self.factor, (tuple, list)):
            factor = rand.uniform(self.factor[0], self.factor[1])
        else:
            factor = self.factor
        ret = self.exec(imgs, factor)
        return ret



class Blur(ABC):
    def __init__(self, radius):
        super(Blur, self).__init__()
        if isinstance(radius, tuple):
            assert radius[1] >= radius[0], 'NEBULAE ERROR ៙ the second element should not be less than the first.'
        elif isinstance(radius, int):
            assert radius>=1, 'NEBULAE ERROR ៙ a valid radius should be larger than one.'
        else:
            raise TypeError('NEBULAE ERROR ៙ a valid radius should be an integer or a tuple.')
        self.radius = radius

    def _blur_np(self, img, radius):
        diameter = 2 * radius + 1
        return cv2.GaussianBlur(img, (diameter, diameter), 0)

    def _blur_pil(self, img, radius):
        return img.filter(ImageFilter.GaussianBlur(radius))

    def call(self, img, factor):
        if self._format == NUMPY:
            return self._blur_np(img, factor)
        elif self._format == PIL:
            return self._blur_pil(img, factor)

    def __call__(self, imgs):
        if isinstance(self.radius, tuple):
            radius = rand.randint(self.radius[0], self.radius[1])
        else:
            radius = self.radius
        ret = self.exec(imgs, radius)
        return ret


class Sharpen(ABC):
    def __init__(self, factor):
        super(Sharpen, self).__init__()
        if isinstance(factor, tuple):
            assert factor[1] >= factor[0], 'NEBULAE ERROR ៙ the second element should not be less than the first.'
        elif isinstance(factor, (int, float)):
            assert factor>1, 'NEBULAE ERROR ៙ a valid factor should be larger than one.'
        else:
            raise TypeError('NEBULAE ERROR ៙ a valid factor should be an float or a tuple.')
        self.factor = factor

    def _sharpen_np(self, img, factor):
        radius = min(3, int(factor))
        diameter = 2 * radius + 1
        blurred = cv2.GaussianBlur(img, (diameter, diameter), 0)
        return cv2.addWeighted(img, factor, blurred, 1-factor, 0)

    def _sharpen_pil(self, img, factor):
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)

    def call(self, img, factor):
        if self._format == NUMPY:
            return self._sharpen_np(img, factor)
        elif self._format == PIL:
            return self._sharpen_pil(img, factor)

    def __call__(self, imgs):
        if isinstance(self.factor, tuple):
            factor = self.factor[0] + rand.random()*(self.factor[1]-self.factor[0])
        else:
            factor = self.factor
        ret = self.exec(imgs, factor)
        return ret



class Noise(ABC):
    def __init__(self, theta, distrib):
        super(Noise, self).__init__()
        self.theta = theta
        assert distrib in (GAUSSIAN, POISSON), 'NEBULAE ERROR ៙ the distribution is either to be Gaussian or Poisson.'
        self.distrib = distrib

    def _noise_np(self, img, theta):
        h, w, c = img.shape
        if self.distrib == GAUSSIAN:
            noise = (np.random.normal(scale=self.theta, size=(h, w, c)) * 255).astype(np.uint8)
        else:
            noise = np.random.poisson(img * theta).astype(np.uint8)
        img = cv2.addWeighted(img, 1, noise, 1, 0)
        return img

    def _noise_pil(self, img, theta):
        img = np.array(img)
        img = self._noise_np(img, theta)
        img = Image.fromarray(img)
        return img

    def call(self, img, theta):
        if self._format == NUMPY:
            return self._noise_np(img, theta)
        elif self._format == PIL:
            return self._noise_pil(img, theta)

    def __call__(self, imgs):
        if isinstance(self.theta, tuple):
            theta = self.theta[0] + rand.random()*(self.theta[1]-self.theta[0])
        else:
            theta = self.theta
        ret = self.exec(imgs, theta)
        return ret



class Sinc(ABC):
    def __init__(self, omega, diameter, ret_img=True):
        # >| omega: usually between (0, pi]
        super(Sinc, self).__init__()
        self.omega = omega
        self.diameter = diameter
        self.ret_img = ret_img

    def _sinc_np(self, img, omega, diameter):
        cx = cy = (diameter - 1) / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            kernel = np.fromfunction(lambda x,y :
                                     omega*special.j1(omega*np.sqrt((x-cx)**2 + (y-cy)**2))
                                     / (2*np.pi*np.sqrt((x-cx)**2 + (y-cy)**2)),
                                        [diameter, diameter])
        if diameter % 2:
            kernel[(diameter-1)//2, (diameter-1)//2] = omega**2 / (4*np.pi)
        kernel /= np.sum(kernel)
        if self.ret_img:
            img = img.astype(np.float32)
            if img.ndim == 3:
                arr = np.zeros_like(img)
                for c in range(img.shape[-1]):
                    arr[:, :, c] = signal.convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
                img = np.clip(arr, 0, 255)
            else:
                img = np.clip(signal.convolve2d(img, kernel, mode='same', boundary='symm'), 0, 255)
            return img.astype(np.uint8)
        else:
            if img.ndim == 3:
                return np.tile(np.expand_dims(kernel, -1), (1, 1, 3))
            else:
                return kernel

    def _sinc_pil(self, img, omega, diameter):
        img = np.array(img)
        img = self._sinc_np(img, omega, diameter)
        if self.ret_img:
            dst = Image.fromarray(img)
        else:
            dst = img
        return dst

    def call(self, img, omega, diameter):
        if self._format == NUMPY:
            return self._sinc_np(img, omega, diameter)
        elif self._format == PIL:
            return self._sinc_pil(img, omega, diameter)

    def __call__(self, imgs):
        if isinstance(self.omega, tuple):
            omega = rand.uniform(self.omega[0], self.omega[1])
        else:
            omega = self.omega
        if isinstance(self.diameter, tuple):
            diameter = rand.randint(self.diameter[0], self.diameter[1])
        else:
            diameter = self.diameter
        ret = self.exec(imgs, omega, diameter)
        return ret



class JPEG(ABC):
    def __init__(self, quality):
        super(JPEG, self).__init__()
        if isinstance(quality, tuple):
            assert quality[1] >= quality[0], 'NEBULAE ERROR ៙ the second element should not be less than the first.'
            assert quality[0]>0 and quality[1]<100, 'NEBULAE ERROR ៙ a valid quality should be an integer within [1, 99].'
        elif isinstance(quality, int):
            assert quality>0 and quality<100, 'NEBULAE ERROR ៙ a valid quality should be an integer within [1, 99].'
        else:
            raise ValueError('NEBULAE ERROR ៙ a valid quality should be an integer or a tuple.')
        self.quality = quality

    def _jpeg_np(self, img, qlt):
        _, img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, qlt])
        dst = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        return dst

    def _jpeg_pil(self, img, qlt):
        buffer = BytesIO()
        img.save(buffer, 'JPEG', quality=qlt)
        return Image.open(buffer)

    def call(self, img, qlt):
        if self._format == NUMPY:
            return self._jpeg_np(img, qlt)
        elif self._format == PIL:
            return self._jpeg_pil(img, qlt)

    def __call__(self, imgs):
        if isinstance(self.quality, tuple):
            qlt = int(rand.randint(self.quality[0], self.quality[1]) * 0.96 + 0.1)
        else:
            qlt = int(self.quality * 0.96 + 0.1)
        ret = self.exec(imgs, qlt)
        return ret



class WebP(ABC):
    def __init__(self, quality):
        super(WebP, self).__init__()
        if isinstance(quality, tuple):
            assert quality[1] >= quality[0], 'NEBULAE ERROR ៙ the second element should not be less than the first.'
            assert quality[0]>0 and quality[1]<100, 'NEBULAE ERROR ៙ a valid quality should be an integer within [1, 99].'
        elif isinstance(quality, int):
            assert quality>0 and quality<100, 'NEBULAE ERROR ៙ a valid quality should be an integer within [1, 99].'
        else:
            raise ValueError('NEBULAE ERROR ៙ a valid quality should be an integer or a tuple.')
        self.quality = quality

    def _webp_np(self, img, qlt):
        _, img = cv2.imencode('.webp', img, [cv2.IMWRITE_WEBP_QUALITY, qlt])
        dst = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        return dst

    def _webp_pil(self, img, qlt):
        buffer = BytesIO()
        img.save(buffer, 'WebP', quality=qlt)
        return Image.open(buffer)

    def call(self, img, qlt):
        if self._format == NUMPY:
            return self._png_np(img, qlt)
        elif self._format == PIL:
            return self._png_pil(img, qlt)

    def __call__(self, imgs):
        if isinstance(self.quality, tuple):
            qlt = rand.randint(self.quality[0], self.quality[1])
        else:
            qlt = self.quality
        ret = self.exec(imgs, qlt)
        return ret



def _vid_encode(imgs, fps, bitrate, codec, _format):
    buf = BytesIO()
    with av.open(buf, 'w', 'mp4') as container:
        stream = container.add_stream(codec, rate=fps)
        if _format == NUMPY:
            h, w = imgs[0].shape[:2]
        elif _format == PIL:
            w, h = imgs[0].size
        else:
            raise Exception('NEBULAE ERROR ៙ the image format has not been assigned.')
        stream.height = h
        stream.width = w
        stream.pix_fmt = 'yuv420p'
        stream.bit_rate = bitrate

        for img in imgs:
            if _format == NUMPY:
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            elif _format == PIL:
                frame = av.VideoFrame.from_image(img)
            frame.pict_type = 'NONE'
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

    outputs = []
    with av.open(buf, 'r', 'mp4') as container:
        if container.streams.video:
            if _format == NUMPY:
                for frame in container.decode(**{'video': 0}):
                    outputs.append(frame.to_rgb().to_ndarray())
            else:
                for frame in container.decode(**{'video': 0}):
                    outputs.append(frame.to_rgb().to_image())
    return outputs



class VABC(ABC):
    def __init__(self, fps, br):
        super(VABC, self).__init__()
        self.fps = fps
        self.br = br

    def __call__(self, seqs):
        if isinstance(self.fps, tuple):
            fps = rand.randint(self.fps[0], self.fps[1])
        else:
            fps = self.fps
        if isinstance(self.br, tuple):
            br = rand.randint(self.br[0], self.br[1])
        else:
            br = self.br

        if isinstance(seqs[0], abc.Sequence):
            ret = []
            for s in seqs:
                ret.append(self.call(s, fps, br))
        else:
            ret = self.call(seqs, fps, br)

        return ret



class MPEG4(VABC):
    def __init__(self, fps, br):
        super(MPEG4, self).__init__(fps, br)

    def call(self, imgs, fps, br):
        return _vid_encode(imgs, fps, br, 'mpeg4', self._format)



class H264(VABC):
    def __init__(self, fps, br):
        super(H264, self).__init__(fps, br)

    def call(self, imgs, fps, br):
        return _vid_encode(imgs, fps, br, 'h264', self._format)



class VP9(VABC):
    def __init__(self, fps, br):
        super(VP9, self).__init__(fps, br)

    def call(self, imgs, fps, br):
        return _vid_encode(imgs, fps, br, 'vp9', self._format)



class AV1(VABC):
    def __init__(self, fps, br):
        super(AV1, self).__init__(fps, br)

    def call(self, imgs, fps, br):
        return _vid_encode(imgs, fps, br, 'av1', self._format)



if __name__ == '__main__':
    from time import time
    import cv2
    from os.path import join as pjoin
    from PIL import ImageEnhance
    DROOT = '/Users/Seria/Desktop/nebulae/test'
    # >| VABC test
    seq = []
    for i in range(5):
        img = cv2.imread('/vid/%04d.png'%(i+1))
        img = img[:,:,::-1]
        seq.append(img)
    cbr = Comburant(H264(30, 6e5))
    dst = cbr(seq)
    for i in range(5):
        # dst[i].save(pjoin(DROOT, 'img/%04d.png'%(i+1)), format='PNG', compress_level=0)
        cv2.imwrite(pjoin(DROOT, 'img/np-lr-%03d.png' % (i + 1)),
                    dst[i][:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])


    # >| ABC test
    cbr = Comburant(Crop((128, 128), central=True, pair_fn=multiple((1, 2))),
                    Resize((100, 100), pair_fn=multiple((1, 2))),
                    Flip(HORIZONTAL),
                    Rotate(30, True),
                    Brighten(1.2),
                    Contrast(1.5),
                    Saturate(0.6),
                    Hue(0.3),
                    Blur(1),
                    Sharpen(1.5),
                    Noise(0.2, POISSON),
                    Sinc(np.pi/3, 15),
                    JPEG(90),
                    )#format=PIL, end=False)
    t_ = time()
    niter = 20
    lrs = []
    hrs = []
    for i in range(niter):
        path = pjoin(DROOT, 'vid/%04d.png'%(i+1))

        # img = Image.open(path)
        # hrs.append(img)
        # lrs.append(img.resize((img.size[0] // 2, img.size[1] // 2)))

        img = cv2.imread(path)
        img = img[:,:,::-1]
        hrs.append(img)
        lrs.append(cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)))

    imgs = cbr([lrs, hrs])
    _t = time()
    for i in range(1):
        # imgs[0][i].save(pjoin(DROOT, 'img/pil-lr-%03d.png' % (i + 1)), format='PNG', compress_level=0)
        # imgs[1][i].save(pjoin(DROOT, 'img/pil-hr-%03d.png' % (i + 1)), format='PNG', compress_level=0)
        cv2.imwrite(pjoin(DROOT, 'img/np-lr-%03d.png'%(i+1)), imgs[0][i][:,:,::-1])#, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(pjoin(DROOT, 'img/np-hr-%03d.png'%(i+1)), imgs[1][i][:,:,::-1])#, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print((_t-t_)/niter)