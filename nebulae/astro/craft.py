#!/usr/bin/env python
'''
component_pt
Created by Seria at 05/02/2019 1:41 PM
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
import os
from math import ceil, sqrt
from collections.abc import Iterator
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..kit import ver2num
from ..rule import ENV_RANK

__all__ = ('Craft',
           
           'FP16', 'INT16', 'INT8',
           'NEAREST', 'LINEAR', 'CUBIC', 'AREA',
           'CONSTANT', 'REFLECT', 'BORDER',

           'Void', 'XavierNorm', 'XavierUnif', 'Normal', 'Uniform', 'Orthog', 'Zeros', 'Ones',
           'Conv', 'TransConv', 'GraphConvEig', 'GraphConvLap', 'GraphConvAdj', 'Dense', 'Embed', 'Pad', 'Identity',

           'RNN', 'BiRNN', 'LSTM', 'BiLSTM', 'MHAttn',

           'Mean', 'Max', 'Min', 'Argmax', 'Argmin', 'Maxele', 'Minele', 'Sum', 'Dot', 'Scatter', 'Gather', 'Grad',

           'QR', 'Eigen', 'MaxEigen', 'FFT', 'IFFT',

           'Roll', 'Reshape', 'Permute', 'Expand', 'Squash', 'Tile', 'Zoom', 'SubPix', 'SurPix', 'GridSample', 'Warp', 
           'MaxPool', 'AvgPool',

           'EMA', 'Retroact',

           'Sqrt', 'Exp', 'Log', 'Sin', 'Cos',

           'Concat', 'Stack',

           'D2S', 'S2D', 'SpDot',

           'Clip', 'Dropout', 'BN', 'CBN', 'IN', 'CIN', 'LN', 'GN', 'SN',

           'Relu', 'LRelu', 'PRelu', 'Gelu', 'Silu', 'Tanh', 'Sigm', 'Sftm', 'Sftp',

           'MAE', 'MSE', 'Huber', 'Charbon', 'SigmXE', 'SftmXE', 'OHEM',

           'AccCls', 'PSNR', 'SSIM',

           'StepLR', 'PolyLR', 'CosLR', 'ExpLR', 'WavyLR',
           'Momentum', 'Nesterov', 'RMSProp', 'Adam', 'AdamW', 'Lion', 'Lamb')



NEAREST = 0
LINEAR = 1
CUBIC = 2
AREA = 3
PT_INTERP = ({NEAREST: 'nearest', LINEAR: 'linear', AREA: 'area'},
             {LINEAR: 'bilinear', CUBIC: 'bicubic', AREA: 'area'},
             {LINEAR: 'trilinear', AREA: 'area'})
SAMPLE_INTERP = {NEAREST: 'nearest', LINEAR: 'bilinear', CUBIC: 'bicubic'}

CONSTANT = 10
REFLECT = 11
BORDER = 12
PT_PAD = {CONSTANT: 'constant', REFLECT: 'reflect', BORDER: 'replicate'}
SAMPLE_PAD = {CONSTANT: 'zeros', REFLECT: 'reflection', BORDER: 'border'}

FREE = 20
DYNAMIC = 21
STATIC = 22

FP16 = torch.float16
INT16 = torch.int16
INT8 = torch.int8

PT_VER = ver2num(torch.__version__)



class Craft(nn.Module):
    def __init__(self, scope='CRAFT'):
        super(Craft, self).__init__()
        self.scope = scope
        self.__prec = None
        self.__dict = {}

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        if self.__prec is None:
            return self.run(*args, **kwargs)
        else:
            with torch.autocast(device_type='cuda', dtype=self.__prec):
                return self.run(*args, **kwargs)

    def lowp(self, prec=FP16):
        assert prec in (FP16, ), 'NEBULAE ERROR ៙ %s is not a valid type of tensors.' % prec
        self.__prec = prec
        self.half()

    def mixp(self, prec=FP16):
        assert prec in (FP16, ), 'NEBULAE ERROR ៙ %s is not a valid type of tensors.' % prec
        self.__prec = prec

    # def gear(self, gr):
    #     if isinstance(gr, bool):
    #         if gr:
    #             self.train()
    #         else:
    #             self.eval()
    #     elif isinstance(gr, Engine):
    #         if gr.gearbox == DYNAMIC:
    #             torch.set_float32_matmul_precision('high')
    #             self = torch.compile(self, dynamic=True)
    #         elif gr.gearbox == STATIC:
    #             torch.set_float32_matmul_precision('high')
    #             self = torch.compile(self)

    #         if gr.device == GPU:
    #             if gr.rank < 0:
    #                 if gr.multi_piston:
    #                     self = DP(self) # [self] is not this object anymore, but we'll receive it outside
    #                 self.cuda()
    #             else:
    #                 self.to(gr.chip[gr.rank])
    #         elif gr.device == CPU:
    #             self.cpu()
    #     else:
    #         raise Exception('NEBULAE ERROR ៙ %s is not a valid type of gear.' % type(gr))
    #     return self

    def vars(self):
        return self.parameters()

    def weights(self):
        raise Exception('NEBULAE ERROR ៙ only a few crafts have weights.')

    def __getitem__(self, key):
        paths = key.split('/')
        craft = self
        for p in paths[:-1]:
            craft = getattr(craft, p)
        if paths[-1] == '':
            return craft
        else:
            return craft.__dict.get(paths[-1])

    def __setitem__(self, key, value):
        self.__dict[key] = value



# -------------------------------------- Layer --------------------------------------- #

class Void(object):
    def __init__(self):
        pass

    def __call__(self, params):
        pass



class XavierNorm(object):
    def __init__(self, scale=1):
        self.s = scale
        self.iniz = nn.init.xavier_normal_

    def __call__(self, params):
        self.iniz(params)
        if self.s != 1:
            params.data *= self.s



class XavierUnif(object):
    def __init__(self, scale=1):
        self.s = scale
        self.iniz = nn.init.xavier_uniform_

    def __call__(self, params):
        self.iniz(params)
        if self.s != 1:
            params.data *= self.s



class Normal(object):
    def __init__(self, mean=0, std=1, scale=1):
        self.mean = mean
        self.std = std
        self.s = scale
        self.iniz = nn.init.normal_

    def __call__(self, params):
        self.iniz(params, self.mean, self.std)
        if self.s != 1:
            params.data *= self.s



class Uniform(object):
    def __init__(self, min=-1, max=1, scale=1):
        self.min_val = min
        self.max_val = max
        self.s = scale
        self.iniz = nn.init.uniform_

    def __call__(self, params):
        self.iniz(params, a=self.min_val, b=self.max_val)
        if self.s != 1:
            params.data *= self.s



class Orthog(object):
    def __init__(self, scale=1):
        self.s = scale
        self.iniz = nn.init.orthogonal_

    def __call__(self, params):
        self.iniz(params)
        if self.s != 1:
            params.data *= self.s



class Ones(object):
    def __init__(self, scale=1):
        self.s = scale
        self.iniz = nn.init.ones_

    def __call__(self, params):
        self.iniz(params)
        if self.s != 1:
            params.data *= self.s



class Zeros(object):
    def __init__(self):
        self.iniz = nn.init.zeros_

    def __call__(self, params):
        self.iniz(params)



class Conv(Craft):
    def __init__(self, in_chs, out_chs, kernel: tuple, stride=1, padding=0, dilation=1, group=1,
                 w_init=XavierNorm(), b_init=Zeros(), scope='CONV'):
        '''
        Args:
        - in_chs: input channel
        - out_chs: output channel
        - kernel: kernel size (must be a tuple)
        - stride: moving stride
        - padding: padding size
        - dilation: stride in atrous convolution
        - group: number of groups to be divided
        - w_init: weight initializer
        - w_param: options for initializing weight
        - b_init: bias initializer
        - b_param: options for initializing bias
        - scope: name scope
        '''
        super(Conv, self).__init__(scope)
        dim = len(kernel)
        custom = False
        if isinstance(w_init, torch.Tensor):
            weight = w_init
            custom = True
            if isinstance(b_init, torch.Tensor):
                bias = b_init
            else:
                bias = None

        if custom:
            if dim == 1:
                conv_fn = F.conv1d
            elif dim == 2:
                conv_fn = F.conv2d
            elif dim == 3:
                conv_fn = F.conv3d
            else:
                raise Exception('NEBULAE ERROR ៙ %d-d convolution is not supported.' % dim)

            if isinstance(stride, int):
                stride = dim * [stride]
            if isinstance(dilation, int):
                dilation = dim * [dilation]

            if isinstance(padding, (tuple, list)) and len(padding) == 2 * dim:
                self.padding = padding
                self.conv = partial(conv_fn, weight=weight, bias=bias, stride=stride, dilation=dilation, groups=group)
            else:
                self.conv = partial(conv_fn, weight=weight, bias=bias, stride=stride, padding=padding,
                                    dilation=dilation, groups=group)
        else:
            if dim == 1:
                conv_fn = nn.Conv1d
            elif dim == 2:
                conv_fn = nn.Conv2d
            elif dim == 3:
                conv_fn = nn.Conv3d
            else:
                raise Exception('NEBULAE ERROR ៙ %d-d convolution is not supported.' % dim)

            if isinstance(stride, int):
                stride = dim * [stride]
            if isinstance(dilation, int):
                dilation = dim * [dilation]

            if isinstance(padding, (tuple, list)) and len(padding) == 2*dim:
                self.padding = padding
                self.conv = conv_fn(in_chs, out_chs, kernel, stride=stride, dilation=dilation,
                                    groups=group, bias=False if isinstance(b_init, Void) else True)
            else:
                self.conv = conv_fn(in_chs, out_chs, kernel, stride=stride, padding=padding, dilation=dilation,
                                    groups=group, bias=False if isinstance(b_init, Void) else True)
            w_init(self.conv.weight)
            b_init(self.conv.bias)

    def weights(self):
        return self.conv.weight

    def run(self, x):
        if hasattr(self, 'padding'):
            x = F.pad(x, self.padding)
        y = self.conv(x)
        return y



class TransConv(Craft):
    def __init__(self, in_chs, out_chs, out_size, kernel: tuple, stride=1, padding=0, dilation=1, group=1,
                 w_init=XavierNorm(), b_init=Zeros(), scope='TRANSCONV'):
        '''
        Args:
        - in_chs: input channel
        - out_chs: output channel
        - out_size: output size
        - kernel: kernel size (must be a tuple)
        - stride: moving stride
        - padding: padding size
        - dilation: stride in atrous convolution
        - group: number of groups to be divided
        - w_init: weight initializer
        - w_param: options for initializing weight
        - b_init: bias initializer
        - b_param: options for initializing bias
        - scope: name scope
        '''
        super(TransConv, self).__init__(scope)
        dim = len(kernel)
        if dim == 1:
            conv_fn = nn.ConvTranspose1d
        elif dim == 2:
            conv_fn = nn.ConvTranspose2d
        elif dim == 3:
            conv_fn = nn.ConvTranspose3d
        else:
            raise Exception('NEBULAE ERROR ៙ %d-d convolution is not supported.' % dim)

        if isinstance(stride, int):
            stride = dim * [stride]
        if isinstance(dilation, int):
            dilation = dim * [dilation]
        if isinstance(padding, int):
            padding = dim * [padding]
            # compensation = dim * [0]
        elif isinstance(padding, (list, tuple)):
            pad = padding
            padding = []
            # compensation = []
            for d in range(dim-1, -1, -1):
                n_elm = pad[2*d] + pad[2*d+1]
                half_elm = ceil(n_elm / 2)
                padding.append(half_elm)
                # compensation.append(n_elm%2)

        self.target_shape = (out_chs,) + out_size
        self.conv = conv_fn(in_chs, out_chs, kernel, stride=stride, padding=padding,
                            dilation=dilation, groups=group, bias=False if isinstance(b_init, Void) else True)
        w_init(self.conv.weight)
        b_init(self.conv.bias)

    def weights(self):
        return self.conv.weight

    def run(self, x):
        target_shape = (x.shape[0],) + self.target_shape
        y = self.conv(x, output_size=target_shape)
        return y



class GraphConvEig(Craft):
    def __init__(self, in_chs, out_chs, eigm, w_init=XavierNorm(), b_init=Zeros(), scope='GRAPHCONVEIG'):
        super(GraphConvEig, self).__init__(scope)
        self.eigm = eigm
        self.weight = nn.Parameter(torch.Tensor(eigm.shape[-1], 1))
        self.proj = nn.Linear(in_chs, out_chs, False)
        if not isinstance(b_init, Void):
            self.bias = nn.Parameter(torch.Tensor(out_chs))
        else:
            self.register_parameter('bias', None)

        self.iniz = (w_init, b_init)
        self.reset_parameters()

    def reset_parameters(self):
        self.iniz[0](self.weight)
        self.iniz[0](self.proj.weight)
        if self.bias is not None:
            self.iniz[1](self.bias)

    def run(self, x):
        support = self.eigm.T @ self.proj(x)
        y = self.weight * support
        y = self.eigm @ y
        if self.bias is not None:
            y += self.bias
        return y

class GraphConvLap(Craft):
    def __init__(self, order, in_chs, out_chs, lap, w_init=XavierNorm(), b_init=Zeros(), scope='GRAPHCONVLAP'):
        super(GraphConvLap, self).__init__(scope)
        self.k = order
        neighb = [torch.eye(lap.shape[0], dtype=lap.dtype)]
        for _ in range(self.k-1):
            neighb.append(neighb[-1] @ lap)
        self.weight = nn.Parameter(torch.Tensor(self.k, 1, 1))
        self.proj = nn.Linear(in_chs, out_chs, False)
        if not isinstance(b_init, Void):
            self.bias = nn.Parameter(torch.Tensor(out_chs))
        else:
            self.register_parameter('bias', None)
        self.neighb = torch.stack(neighb)

        self.iniz = (w_init, b_init)
        self.reset_parameters()

    def reset_parameters(self):
        self.iniz[0](self.weight)
        self.iniz[0](self.proj.weight)
        if self.bias is not None:
            self.iniz[1](self.bias)

    def run(self, x):
        support = self.proj(x)
        y = torch.sum(self.weight * self.neighb, dim=0) @ support
        if self.bias is not None:
            y += self.bias
        return y

class GraphConvAdj(Craft):
    def __init__(self, in_chs, out_chs, adj, w_init=XavierNorm(), b_init=Zeros(), scope='GRAPHCONVADJ'):
        super(GraphConvAdj, self).__init__(scope)
        self.adj = adj
        self.weight = nn.Parameter(torch.Tensor(in_chs, out_chs))
        if not isinstance(b_init, Void):
            self.bias = nn.Parameter(torch.Tensor(out_chs))
        else:
            self.register_parameter('bias', None)

        self.iniz = (w_init, b_init)
        self.reset_parameters()

    def reset_parameters(self):
        self.iniz[0](self.weight)
        if self.bias is not None:
            self.iniz[1](self.bias)

    def run(self, x):
        support = torch.mm(x, self.weight)
        y = torch.sparse.mm(self.adj, support)
        if self.bias is not None:
            y += self.bias
        return y



class Dense(Craft):
    def __init__(self, in_chs, out_chs, axis=-1,
                 w_init=XavierNorm(), b_init=Zeros(), scope='DENSE'):
        super(Dense, self).__init__(scope)
        if axis == 0:
            raise Exception('NEBULAE ERROR ៙ you cannot apply dense layer along batch axis.')
        else:
            self.axis = axis

        self.fc = nn.Linear(in_chs, out_chs, bias=False if isinstance(b_init, Void) else True)
        w_init(self.fc.weight)
        b_init(self.fc.bias)

    def weights(self):
        return self.fc.weight

    def run(self, x):
        if self.axis == -1:
            y = self.fc(x)
        else:
            dim = x.ndim()
            permuted = [i for i in range(dim)]
            permuted = permuted[:self.axis] + permuted[self.axis + 1:] + [self.axis]
            x = x.transpose(*permuted)
            y = self.fc(x)
            permuted = [i for i in range(dim)]
            permuted = permuted[:self.axis] + [dim - 1] + permuted[self.axis:-1]
            y = y.transpose(*permuted)
        return y



class Embed(Craft):
    def __init__(self, ntoken, token_dim, scope='EMBED'):
        super(Embed, self).__init__(scope)
        self.embd = nn.Embedding(ntoken, token_dim)

    def weights(self):
        return self.embd.weight

    def run(self, x):
        y = self.embd(x)
        return y



class Pad(Craft):
    def __init__(self, padding, mode=CONSTANT, value=0., scope='PAD'):
        super(Pad, self).__init__(scope)
        self.padding = padding
        self.mode = PT_PAD[mode]
        self.value = value

    def run(self, x):
        y = nn.functional.pad(x, self.padding, self.mode, self.value)
        return y



class Identity(Craft):
    def __init__(self, scope='IDENTITY'):
        super(Identity, self).__init__(scope)

    def run(self, x):
        return x



# ---------------------------------- Manipulation ------------------------------------ #

class EMA(Craft):
    def __init__(self, hull, decay_fn=lambda x: 0.9, on_device=False, scope='EMA'):
        super(EMA, self).__init__(scope)
        if isinstance(hull, (torch.nn.DataParallel, torch.nn.parallel.distributed.DistributedDataParallel)):
            hull = hull.module
        self.counter = 0
        self.decay_fn = decay_fn
        self.on_device = on_device
        self._rank = int(os.environ.get(ENV_RANK, -1))
        self['hull'] = hull
        self.swapped = False # whether have swapped to its shadow

        # initialize shadow as hull itself
        self.shadow = {}
        hull_params = hull.state_dict()
        # note that params have not been to GPU even cuda is enabled
        for k, v in hull_params.items():
            self.shadow[k] = v.clone().detach()
        # assume that all params in hull are on the same device
        self.hull_dev = v.device
        print('+---------------------------------------------+')
        print('| The EMAed parameters are detected on %s |'%str(self.hull_dev))
        print('+---------------------------------------------+')
        if self.on_device:
            self.shadow_dev = self.hull_dev
        else:
            self.shadow_dev = torch.device('cpu')
        for k in self.shadow.keys():
            self.shadow[k] = self.shadow[k].to(self.shadow_dev)
    
    # def gear(self, gr):
    #     if isinstance(gr, torch.device):
    #         self.hull.to(gr)
    #         self.hull_dev = gr
    #     else:
    #         self.hull = self.hull.gear(gr)
    #         if isinstance(gr, Engine):
    #             self.hull_dev = gr.chip[0] # update only take places on main device
    #             if self.on_device:
    #                 self.shadow_dev = self.hull_dev
    #             else:
    #                 self.shadow_dev = torch.device('cpu')
    #     for k in self.shadow.keys():
    #         self.shadow[k] = self.shadow[k].to(self.shadow_dev)
    #     return self

    def vars(self):
        return getattr(self.hull, 'vars', self.hull.parameters)()

    def weights(self):
        return self.hull.weights()
    
    def __getattr__(self, name: str): # only be invoked when getattr failed.
        # EMA is basically a wrapper of model i.e. self.hull,
        # hence we add the following lines to make sure the
        # attributes in the model is accessible.
        hull = self['hull']
        if name == 'hull':
            return hull
        if hasattr(hull, name):
            return getattr(hull, name)

        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def _swap(self):
        if self._rank <= 0:
            with torch.no_grad():
                hull_params = self.hull.state_dict()
                for key in self.shadow.keys():
                    # h' = h+s
                    hull_params[key].add_(self.shadow[key].to(self.hull_dev))
                    # s' = h'-s = h
                    self.shadow[key].data.copy_(hull_params[key].to(self.shadow_dev).data - self.shadow[key].data)
                    # h' = h'-s' = s
                    hull_params[key].sub_(self.shadow[key].to(self.hull_dev))
        self.swapped = not self.swapped

    # >| turn on shadow to evalute
    def on(self):
        if not self.swapped:
            self._swap()

    # >| turn off shadow for training network weights
    def off(self):
        if self.swapped:
            self._swap()

    def update(self):
        if self._rank > 0:
            return
        self.counter += 1 # count calling times
        decay = self.decay_fn(self.counter)
        with torch.no_grad():
            hull_params = self.hull.state_dict()
            for key in hull_params.keys():
                self.shadow[key].data.copy_(
                    hull_params[key].to(self.shadow_dev).data * (1 - decay) + self.shadow[key].data * decay)

    def run(self, *args, **kwargs):
        return self.hull(*args, **kwargs)



class Retroact(Craft):
    def __init__(self, scope='RETROACT'):
        super(Retroact, self).__init__(scope)

    def run(self, i, o):
        i.retain_grad()
        # h = i.register_hook(grabber)
        o.backward() # compute gradients

        pooled_grad = F.adaptive_avg_pool2d(i.grad, (1, 1))

        i *= pooled_grad

        hmap = i.detach()
        hmap = torch.mean(hmap, dim=1, keepdim=True) # average over channels
        hmap.clamp_(0) # pull negative values up to zero
        for b in range(hmap.shape[0]):
            hmap[b, ...] /= torch.max(hmap[b, ...]) + 1e-6

        return hmap



# ----------------------------------- Recurrent -------------------------------------- #

class RNN(Craft):
    def __init__(self, in_chs, hid_chs, nlayers=1, scope='RNN'):
        super(RNN, self).__init__(scope)
        self.rnn = nn.RNN(in_chs, hid_chs, nlayers)

    def run(self, x, h=None):
        y, h = self.rnn(x, h)
        return y, h



class BiRNN(Craft):
    def __init__(self, in_chs, hid_chs, nlayers=1, scope='BIRNN'):
        super(BiRNN, self).__init__(scope)
        self.birnn = nn.RNN(in_chs, hid_chs, nlayers, bidirectional=True)

    def run(self, x, h=None):
        y, h = self.birnn(x, h)
        return y, h



class LSTM(Craft):
    def __init__(self, in_chs, hid_chs, nlayers=1, scope='LSTM'):
        super(LSTM, self).__init__(scope)
        self.lstm = nn.LSTM(in_chs, hid_chs, nlayers)

    def run(self, x, h=None, c=None):
        h_n_c = None if h is None or c is None else (h, c)
        y, h_n_c = self.lstm(x, h_n_c)
        h, c = h_n_c
        return y, h, c



class BiLSTM(Craft):
    def __init__(self, in_chs, hid_chs, nlayers=1, scope='BILSTM'):
        super(BiLSTM, self).__init__(scope)
        self.bilstm = nn.LSTM(in_chs, hid_chs, nlayers, bidirectional=True)

    def run(self, x, h=None, c=None):
        h_n_c = None if h is None or c is None else (h, c)
        y, h_n_c = self.bilstm(x, h_n_c)
        h, c = h_n_c
        return y, h, c



class MHAttn(Craft):
    def __init__(self, nheads, q_chs, k_chs, v_chs, p_drop=0., scope='MHATTN'):
        super(MHAttn, self).__init__(scope)
        self.mha = nn.MultiheadAttention(q_chs, nheads, p_drop, kdim=k_chs, vdim=v_chs)

    def run(self, q, k, v, pad_mask=None, attn_mask=None):
        y, a = self.mha(q, k, v, key_padding_mask=pad_mask, attn_mask=attn_mask)
        return y, a



# ----------------------------------- Mathmatic -------------------------------------- #

class Sqrt(Craft):
    def __init__(self, scope='SQRT'):
        super(Sqrt, self).__init__(scope)

    def run(self, t):
        y = torch.sqrt(t)
        return y



class Exp(Craft):
    def __init__(self, scope='EXP'):
        super(Exp, self).__init__(scope)

    def run(self, t):
        y = torch.exp(t)
        return y



class Log(Craft):
    def __init__(self, scope='LOG'):
        super(Log, self).__init__(scope)

    def run(self, t):
        y = torch.log(t)
        return y



class Sin(Craft):
    def __init__(self, scope='SIN'):
        super(Sin, self).__init__(scope)

    def run(self, t):
        y = torch.sin(t)
        return y



class Cos(Craft):
    def __init__(self, scope='COS'):
        super(Cos, self).__init__(scope)

    def run(self, t):
        y = torch.cos(t)
        return y


# ------------------------------------ Polyadic -------------------------------------- #

class Concat(Craft):
    def __init__(self, scope='CONCAT'):
        super(Concat, self).__init__(scope)

    def run(self, t, axis=-1):
        y = torch.cat(t, dim=axis)
        return y



class Stack(Craft):
    def __init__(self, scope='STACK'):
        super(Stack, self).__init__(scope)

    def run(self, t, axis=-1):
        y = torch.stack(t, dim=axis)
        return y



# ------------------------------------- Sparse --------------------------------------- #

class D2S(Craft):
    def __init__(self, scope='D2S'):
        super(D2S, self).__init__(scope)

    def run(self, x):
        y = x.to_sparse()
        return y



class S2D(Craft):
    def __init__(self, scope='S2D'):
        super(S2D, self).__init__(scope)

    def run(self, x):
        y = x.to_dense()
        return y



class SpDot(Craft):
    def __init__(self, scope='SPDOT'):
        super(SpDot, self).__init__(scope)

    def run(self, x, y, in_batch=False):
        # one and only one of inputs must be sparse tensor
        if x.is_sparse:
            if in_batch:
                z = torch.bmm(x, y)
            else:
                z = torch.sparse.mm(x, y)
        else: # x (dense) @ y (sparse) is not supported in pytorch
            if in_batch:
                z = torch.bmm(y.to_dense().T.to_sparse(), x.T).T
            else:
                z = torch.sparse.mm(y.to_dense().T.to_sparse(), x.T).T
        return z



# ----------------------------------- Statistics ------------------------------------- #

class Clip(Craft):
    def __init__(self, scope='CLIP'):
        super(Clip, self).__init__(scope)

    def run(self, x, ranges, in_place=False):
        if isinstance(ranges, tuple):
            assert len(ranges)==2
        else:
            ranges = (-ranges, ranges)
        if in_place:
            x.clamp_(ranges[0], ranges[1])
            return x
        else:
            return torch.clamp(x, ranges[0], ranges[1])



class Mean(Craft):
    def __init__(self, scope='MEAN'):
        super(Mean, self).__init__(scope)

    def run(self, x, axis=None, reduce=True):
        if axis is None:
            assert reduce, 'NEBULAE ERROR ៙ reduce must be True when axis is not specified.'
            y = torch.mean(x)
        else:
            y = torch.mean(x, dim=axis, keepdim=not reduce)
        return y



class Max(Craft):
    def __init__(self, scope='MAX'):
        super(Max, self).__init__(scope)

    def run(self, x, axis=None, reduce=True):
        if axis is None:
            assert reduce, 'NEBULAE ERROR ៙ reduce must be True when axis is not specified.'
            y = torch.max(x)
        else:
            y = torch.max(x, dim=axis, keepdim=not reduce)
        return y



class Min(Craft):
    def __init__(self, scope='MIN'):
        super(Min, self).__init__(scope)

    def run(self, x, axis=None, reduce=True):
        if axis is None:
            assert reduce, 'NEBULAE ERROR ៙ reduce must be True when axis is not specified.'
            y = torch.min(x)
        else:
            y = torch.min(x, dim=axis, keepdim=not reduce)
        return y



class Argmax(Craft):
    def __init__(self, scope='ARGMAX'):
        super(Argmax, self).__init__(scope)

    def run(self, x, axis=None, reduce=True):
        if axis is None:
            assert reduce, 'NEBULAE ERROR ៙ reduce must be True when axis is not specified.'
            y = torch.argmax(x)
        else:
            y = torch.argmax(x, dim=axis, keepdim=not reduce)
        return y



class Argmin(Craft):
    def __init__(self, scope='ARGMIN'):
        super(Argmin, self).__init__(scope)

    def run(self, x, axis=None, reduce=True):
        if axis is None:
            assert reduce, 'NEBULAE ERROR ៙ reduce must be True when axis is not specified.'
            y = torch.argmin(x)
        else:
            y = torch.argmin(x, dim=axis, keepdim=not reduce)
        return y



class Maxele(Craft):
    def __init__(self, scope='MAXELE'):
        super(Maxele, self).__init__(scope)

    def run(self, x, y):
        z = torch.maximum(x, y)
        return z



class Minele(Craft):
    def __init__(self, scope='MINELE'):
        super(Minele, self).__init__(scope)

    def run(self, x, y):
        z = torch.minimum(x, y)
        return z



class Sum(Craft):
    def __init__(self, scope='SUM'):
        super(Sum, self).__init__(scope)

    def run(self, x, axis=None, reduce=True):
        if axis is None:
            assert reduce, 'NEBULAE ERROR ៙ reduce must be True when axis is not specified.'
            y = torch.sum(x)
        else:
            y = torch.sum(x, dim=axis, keepdim=not reduce)
        return y



class Dot(Craft):
    def __init__(self, scope='DOT'):
        super(Dot, self).__init__(scope)

    def run(self, x, y, axes=(), in_batch=False):
        if len(axes) > 0:
            assert not in_batch, 'NEBULAE ERROR ៙ in_batch is invalid when axes are assigned.'
            z = torch.tensordot(x, y, axes)
        else:
            if in_batch:
                z = torch.bmm(x, y)
            else:
                z = torch.mm(x, y)
        return z



class Scatter(Craft):
    def __init__(self, scope='Scatter'):
        super(Scatter, self).__init__(scope)

    def run(self, x, y, axis, idx, in_place=False):
        if in_place:
            return x.scatter_(axis, idx, y)
        else:
            return x.scatter(axis, idx, y)



class Gather(Craft):
    def __init__(self, scope='GATHER'):
        super(Gather, self).__init__(scope)

    def run(self, x, axis, idx):
        y = torch.gather(x, axis, idx)
        return y



class Grad(Craft):
    def __init__(self, scope='GRAD'):
        super(Grad, self).__init__(scope)

    def run(self, i, o, in_batch=False):
        grad_out_weight = torch.ones((i.shape[0], 1), device=i.device)
        g = torch.autograd.grad(o, i, grad_outputs=grad_out_weight,
                                retain_graph=True, create_graph=True)
        return g



# ---------------------------------- Linear Algebra ---------------------------------- #

class QR(Craft):
    def __init__(self, scope='QR'):
        super(QR, self).__init__(scope)

    def run(self, m, reduced=False):
        q, r = torch.qr(m, some=reduced)
        return q, r



class Eigen(Craft):
    def __init__(self, scope='EIGEN'):
        super(Eigen, self).__init__(scope)

    def run(self, m, in_pair=True):
        eigval, eigvec = torch.eig(m, eigenvectors=in_pair)
        if in_pair:
            return eigval, eigvec
        else:
            return eigval



class MaxEigen(Craft):
    def __init__(self, niters=3, scope='MAXEIGEN'):
        super(MaxEigen, self).__init__(scope)
        self.niters = niters

    def run(self, m, in_pair=True):
        assert (m==m.T).all(), 'NEBULAE ERROR ៙ only when m is a real symmetric matrix can MaxEigen be applied.'
        x = torch.normal(0, 1, size=(m.shape[-1],))
        for _ in range(self.niters):
            x = m @ x
            x /= torch.norm(x) + 1e-10
        eigval_max = x @ m @ x.T
        if not in_pair:
            return eigval_max
        else:
            eigvec_max = (m @ x.T) / eigval_max
            return eigval_max, eigvec_max



class FFT(Craft):
    def __init__(self, dim, eps=1e-10, scope='FFT'):
        super(FFT, self).__init__(scope)
        self.dim = dim
        self.eps = eps
        self.maxv = 1 / eps

    def run(self, m, normal=False, in_pair=True):
        m = torch.stack((m, torch.zeros_like(m)), -1)
        ft = torch.fft(m, self.dim, normalized=normal)
        if in_pair:
            real, imgn = torch.split(ft, 1, dim=-1)
            real = torch.clamp(real, -self.maxv, self.maxv)
            imgn = torch.clamp(imgn, -self.maxv, self.maxv)
            amp = torch.sqrt(real**2 + imgn**2 + self.eps)
            pha = torch.atan2(imgn, real + self.eps)
            return ft, amp, pha
        else:
            return ft



class IFFT(Craft):
    def __init__(self, dim, scope='IFFT'):
        super(IFFT, self).__init__(scope)
        self.dim = dim

    def run(self, m, normal=False):
        ift = torch.ifft(m, self.dim, normalized=False)
        return ift



# ------------------------------------- Resizer -------------------------------------- #

class Roll(Craft):
    def __init__(self, scope='ROLL'):
        super(Roll, self).__init__(scope)

    def run(self, x, shift, axis=None):
        y = torch.roll(x, shift, axis)
        return y



class Reshape(Craft):
    def __init__(self, scope='RESHAPE'):
        super(Reshape, self).__init__(scope)

    def run(self, x, shape):
        y = torch.reshape(x, shape)
        return y



class Permute(Craft):
    def __init__(self, scope='PERMUTE'):
        super(Permute, self).__init__(scope)

    def run(self, x, order):
        y = x.permute(order)
        return y



class Expand(Craft):
    def __init__(self, scope='EXPAND'):
        super(Expand, self).__init__(scope)

    def run(self, x, axis=0):
        y = x.unsqueeze(axis)
        return y



class Squash(Craft):
    def __init__(self, scope='SQUASH'):
        super(Squash, self).__init__(scope)

    def run(self, x, axis=None):
        y = x.squeeze(axis)
        return y



class Tile(Craft):
    def __init__(self, scope='TILE'):
        super(Tile, self).__init__(scope)

    def run(self, x, times):
        y = x.repeat(*times)
        return y



class Zoom(Craft):
    def __init__(self, size: tuple=(), scale: tuple=(), interp=LINEAR, aligned=True, scope='ZOOM'):
        super(Zoom, self).__init__(scope)
        assert len(size)*len(scale)==0 and len(size)+len(scale)>0, \
            'NEBULAE ERROR ៙ either size or scale must be an empty tuple.'
        if len(size)==0:
            self.zoomfn = partial(nn.functional.interpolate, scale_factor=scale)
        else:
            self.zoomfn = partial(nn.functional.interpolate, size=size)
        if interp == AREA or interp == NEAREST:
            self.aligned = None
        else:
            self.aligned = aligned
        self.mode = PT_INTERP[len(scale)-1][interp]

    def run(self, x):
        y = self.zoomfn(x, mode=self.mode, align_corners=self.aligned)
        return y



class SubPix(Craft):
    def __init__(self, scale: tuple, scope='SUBPIX'):
        super(SubPix, self).__init__(scope)
        assert len(scale)==2 and scale[0]==scale[1]
        self.fn = nn.PixelShuffle(scale[0])

    def run(self, x):
        y = self.fn(x)
        return y



class SurPix(Craft):
    def __init__(self, scale: tuple, scope='SURPIX'):
        super(SurPix, self).__init__(scope)
        assert len(scale)==2 and scale[0]==scale[1]
        if PT_VER < ver2num('1.8.0'):
            def rearrange(x):
                S = scale[0]
                C, H, W = x.shape[-3:]
                _C = C * S ** 2
                _H = H // S
                _W = W // S
                d = x.dim()
                indif = tuple(x.shape)[0:-3]
                x = x.reshape(indif + (C, _H, S, _W, S))
                x = x.permute(tuple([i for i in range(d-3)]) + (d-3, d-1, d+1, d-2, d))
                x = x.reshape(indif + (_C, _H, _W))
                return x
            self.fn = rearrange
        else:
            self.fn = nn.PixelUnshuffle(scale[0])

    def run(self, x):
        y = self.fn(x)
        return y



class GridSample(Craft):
    def __init__(self, interp=LINEAR, aligned=True, pad_mode=CONSTANT, scope='GRIDSAMPLE'):
        super(GridSample, self).__init__(scope)
        self.interp = SAMPLE_INTERP[interp]
        self.aligned = aligned
        self.pad_mode = SAMPLE_PAD[pad_mode]

    def run(self, x, grid):
        y = F.grid_sample(x, grid, mode=self.interp, padding_mode=self.pad_mode, align_corners=self.aligned)
        return y



class Warp(Craft):
    def __init__(self, interp=LINEAR, aligned=True, pad_mode=CONSTANT, scope='WARP'):
        super(Warp, self).__init__(scope)
        self.interp = SAMPLE_INTERP[interp]
        self.aligned = aligned
        self.pad_mode = SAMPLE_PAD[pad_mode]

    def run(self, x, flow):
        B, C, H, W = x.shape
        grid_x, grid_y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing='ij')
        grid_x = grid_x.view(1, H, W, 1).repeat(B, 1, 1, 1)
        grid_y = grid_y.view(1, H, W, 1).repeat(B, 1, 1, 1)
        grid = torch.cat((grid_x, grid_y), -1).to(flow.dtype)
        grid.requires_grad = False

        grid += flow.permute(0, 2, 3, 1)
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / max(W - 1, 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / max(H - 1, 1) - 1.0
        y = F.grid_sample(x, grid, mode=self.interp, padding_mode=self.pad_mode, align_corners=self.aligned)
        return y



class MaxPool(Craft):
    def __init__(self, kernel: tuple, stride=2, padding=0, scope='MPOOL'):
        super(MaxPool, self).__init__(scope)
        dim = len(kernel)
        is_global = True if kernel[-1] < 0 else False
        if dim == 1:
            if is_global:
                pool_fn = nn.AdaptiveMaxPool1d
            else:
                pool_fn = nn.MaxPool1d
        elif dim == 2:
            if is_global:
                pool_fn = nn.AdaptiveMaxPool2d
            else:
                pool_fn = nn.MaxPool2d
        elif dim == 3:
            if is_global:
                pool_fn = nn.AdaptiveMaxPool3d
            else:
                pool_fn = nn.MaxPool3d
        else:
            raise Exception('NEBULAE ERROR ៙ %d-d pooling is not supported.' % dim)

        if isinstance(stride, int):
            stride = dim * [stride]

        if isinstance(padding, (tuple, list)) and len(padding) == 2 * dim:
            self.padding = padding
            padding = tuple(dim * [0])
        if is_global:
            assert padding == 0
            self.pool = pool_fn(tuple(dim * [1]))
        else:
            self.pool = pool_fn(kernel_size=kernel, stride=stride, padding=padding)

    def run(self, x):
        if hasattr(self, 'padding'):
            x = nn.functional.pad(x, self.padding, value=-float('inf'))
        y = self.pool(x)
        return y



class AvgPool(Craft):
    def __init__(self, kernel: tuple, stride=2, padding=0, scope='APOOL'):
        super(AvgPool, self).__init__(scope)
        dim = len(kernel)
        is_global = True if kernel[-1]<0 else False
        if dim == 1:
            if is_global:
                pool_fn = nn.AdaptiveAvgPool1d
            else:
                pool_fn = nn.AvgPool1d
        elif dim == 2:
            if is_global:
                pool_fn = nn.AdaptiveAvgPool2d
            else:
                pool_fn = nn.AvgPool2d
        elif dim == 3:
            if is_global:
                pool_fn = nn.AdaptiveAvgPool3d
            else:
                pool_fn = nn.AvgPool3d
        else:
            raise Exception('NEBULAE ERROR ៙ %d-d pooling is not supported.' % dim)

        if isinstance(stride, int):
            stride = dim * [stride]

        if isinstance(padding, (tuple, list)) and len(padding) == 2 * dim:
            self.padding = padding
            padding = tuple(dim * [0])
        if is_global:
            assert padding == 0
            self.pool = pool_fn(tuple(dim * [1]))
        else:
            self.pool = pool_fn(kernel_size=kernel, stride=stride, padding=padding)

    def run(self, x):
        if hasattr(self, 'padding'):
            x = nn.functional.pad(x, self.padding)
        y = self.pool(x)
        return y



# ------------------------------------ Activation ------------------------------------ #

class Relu(Craft):
    def __init__(self, scope='RELU'):
        super(Relu, self).__init__(scope)
        self.actv = nn.ReLU()

    def run(self, x):
        y = self.actv(x)
        return y



class LRelu(Craft):
    def __init__(self, alpha=0.2, scope='LRELU'):
        super(LRelu, self).__init__(scope)
        self.actv = nn.LeakyReLU(alpha)

    def run(self, x):
        y = self.actv(x)
        return y



class PRelu(Craft):
    def __init__(self, alpha=0.2, scope='PRELU'):
        super(PRelu, self).__init__(scope)
        self.actv = nn.PReLU(init=alpha)

    def run(self, x):
        y = self.actv(x)
        return y



class Gelu(Craft):
    def __init__(self, scope='GELU'):
        super(Gelu, self).__init__(scope)
        self.actv = nn.GELU()

    def run(self, x):
        y = self.actv(x)
        return y
    


class Silu(Craft):
    def __init__(self, scope='SILU'):
        super(Silu, self).__init__(scope)
        self.actv = nn.SiLU()

    def run(self, x):
        y = self.actv(x)
        return y



class Tanh(Craft):
    def __init__(self, scope='TANH'):
        super(Tanh, self).__init__(scope)
        self.actv = nn.Tanh()

    def run(self, x):
        y = self.actv(x)
        return y



class Sigm(Craft):
    def __init__(self, scope='SIGM'):
        super(Sigm, self).__init__(scope)
        self.actv = nn.Sigmoid()

    def run(self, x):
        y = self.actv(x)
        return y



class Sftm(Craft):
    def __init__(self, axis=-1, scope='SFTM'):
        super(Sftm, self).__init__(scope)
        self.actv = nn.Softmax(dim=axis)

    def run(self, x):
        y = self.actv(x)
        return y



class Sftp(Craft):
    def __init__(self, scope='SFTP'):
        super(Sftp, self).__init__(scope)
        self.actv = nn.Softplus(1)

    def run(self, x):
        y = self.actv(x)
        return y



# ------------------------------------ Distributing ------------------------------------ #

class Dropout(Craft):
    def __init__(self, p_drop, dim, scope='DROPOUT'):
        super(Dropout, self).__init__(scope)
        if dim == 1:
            dp_fn = nn.Dropout
        elif dim == 2:
            dp_fn = nn.Dropout2d
        elif dim == 3:
            dp_fn = nn.Dropout3d
        else:
            raise Exception('NEBULAE ERROR ៙ %d-d BN is not supported.' % dim)
        self.dp = dp_fn(p=p_drop)

    def run(self, x):
        y = self.dp(x)
        return y



class BN(Craft):
    def __init__(self, out_chs, dim, mmnt=0.9, resilient=True, scope='BN'):
        super(BN, self).__init__(scope)
        if dim == 1:
            norm_fn = nn.BatchNorm1d
        elif dim == 2:
            norm_fn = nn.BatchNorm2d
        elif dim == 3:
            norm_fn = nn.BatchNorm3d
        else:
            raise Exception('NEBULAE ERROR ៙ %d-d BN is not supported.' % dim)
        self.norm = norm_fn(out_chs, momentum=1 - mmnt, affine=resilient, eps=1e-5)

    def weights(self):
        return self.norm.weight

    def run(self, x):
        y = self.norm(x)
        return y



class CBN(Craft):
    def __init__(self, in_chs, out_chs, dim, mmnt=0.9, scope='CBN'):
        super(CBN, self).__init__(scope)
        if dim == 1:
            norm_fn = nn.BatchNorm1d
        elif dim == 2:
            norm_fn = nn.BatchNorm2d
        elif dim == 3:
            norm_fn = nn.BatchNorm3d
        else:
            raise Exception('NEBULAE ERROR ៙ %d-d CN is not supported.' % dim)
        self.norm = norm_fn(out_chs, momentum=1 - mmnt, affine=False, eps=1e-5)
        self.relu = nn.ReLU()
        self.gamma_1 = nn.Linear(in_chs, in_chs // 2)
        self.gamma_2 = nn.Linear(in_chs // 2, out_chs)
        self.beta_1 = nn.Linear(in_chs, in_chs // 2)
        self.beta_2 = nn.Linear(in_chs // 2, out_chs)

    def weights(self):
        return self.weight

    def run(self, x, z):
        y = self.norm(x)

        g = self.gamma_1(z)
        g = self.relu(g)
        g = self.gamma_2(g)

        b = self.beta_1(z)
        b = self.relu(b)
        b = self.beta_2(b)

        for _ in range(x.ndim - 2):
            g = g.unsqueeze(-1)
            b = b.unsqueeze(-1)

        self.weight = g
        self.bias = b
        y = self.weight * y + self.bias

        return y



class IN(Craft):
    def __init__(self, out_chs, dim, mmnt=0.9, resilient=True, scope='IN'):
        super(IN, self).__init__(scope)
        if dim == 1:
            norm_fn = nn.InstanceNorm1d
        elif dim == 2:
            norm_fn = nn.InstanceNorm2d
        elif dim == 3:
            norm_fn = nn.InstanceNorm3d
        else:
            raise Exception('NEBULAE ERROR ៙ %d-d IN is not supported.' % dim)
        self.norm = norm_fn(out_chs, momentum=1 - mmnt, affine=resilient, eps=1e-5)

    def weights(self):
        return self.norm.weight

    def run(self, x):
        y = self.norm(x)
        return y



class CIN(Craft):
    def __init__(self, in_chs, out_chs, dim, mmnt=0.9, scope='CIN'):
        super(CIN, self).__init__(scope)
        if dim == 1:
            norm_fn = nn.InstanceNorm1d
        elif dim == 2:
            norm_fn = nn.InstanceNorm2d
        elif dim == 3:
            norm_fn = nn.InstanceNorm3d
        else:
            raise Exception('NEBULAE ERROR ៙ %d-d CN is not supported.' % dim)
        self.norm = norm_fn(out_chs, momentum=1 - mmnt, affine=False, eps=1e-5)
        self.relu = nn.ReLU()
        self.gamma_1 = nn.Linear(in_chs, in_chs // 2)
        self.gamma_2 = nn.Linear(in_chs // 2, out_chs)
        self.beta_1 = nn.Linear(in_chs, in_chs // 2)
        self.beta_2 = nn.Linear(in_chs // 2, out_chs)

    def weights(self):
        return self.weight

    def run(self, x, z):
        y = self.norm(x)

        g = self.gamma_1(z)
        g = self.relu(g)
        g = self.gamma_2(g)

        b = self.beta_1(z)
        b = self.relu(b)
        b = self.beta_2(b)

        for _ in range(x.ndim - 2):
            g = g.unsqueeze(-1)
            b = b.unsqueeze(-1)

        self.weight = g
        self.bias = b
        y = self.weight * y + self.bias

        return y



class LN(Craft):
    def __init__(self, norm_shape, resilient=True, scope='LN'):
        super(LN, self).__init__(scope)
        if isinstance(norm_shape, tuple) and len(norm_shape)>1:
            norm_shape = tuple([norm_shape[-1]] + [ns for ns in norm_shape[:-1]])
        self.norm = nn.LayerNorm(norm_shape, elementwise_affine=resilient, eps=1e-5)

    def weights(self):
        return self.norm.weight

    def run(self, x):
        y = self.norm(x)
        return y



class GN(Craft):
    def __init__(self, out_chs, ngroup, resilient=True, scope='GN'):
        super(GN, self).__init__(scope)
        self.norm = nn.GroupNorm(ngroup, out_chs, affine=resilient, eps=1e-5)

    def weights(self):
        return self.norm.weight

    def run(self, x):
        y = self.norm(x)
        return y



class SN(Craft):
    def __init__(self, craft, niters=3, eps=1e-12, pname='weight', scope='SN'):
        super(SN, self).__init__(scope)
        self.craft = craft
        self.niters = niters
        self.eps = eps
        self.pname = pname
        if isinstance(craft, (Conv, TransConv)):
            # self.hull = craft.conv
            self.craft.conv = nn.utils.spectral_norm(self.craft.conv, self.pname, self.niters, self.eps)
        elif isinstance(craft, Dense):
            # self.hull = craft.fc
            self.craft.fc = nn.utils.spectral_norm(self.craft.fc, self.pname, self.niters, self.eps)
        elif isinstance(craft, Embed):
            # self.hull = craft.embd
            self.craft.embd = nn.utils.spectral_norm(self.craft.embd, self.pname, self.niters, self.eps)
        elif isinstance(craft, (BN, IN, LN)):
            # self.hull = craft.norm
            self.craft.norm = nn.utils.spectral_norm(self.craft.norm, self.pname, self.niters, self.eps)
        elif isinstance(craft, (CBN, CIN)):
            # self.hull = craft
            self.craft = nn.utils.spectral_norm(self.craft, self.pname, self.niters, self.eps)
        elif isinstance(craft, nn.Module):
            # self.hull = craft
            self.craft = nn.utils.spectral_norm(self.craft, self.pname, self.niters, self.eps)
        else:
            raise Exception('NEBULAE ERROR ៙ SN does not support %s layer.' % type(craft))

    '''
        self.hull = craft[self.key]
        assert not hasattr(self.hull, self.pname + "_u")
        self._register()

    def _normalize(self, v):
        return v / (v.norm() + self.eps)

    def _updateUV(self):
        w = getattr(self.hull, self.pname)
        u = getattr(self.hull, self.pname + "_u")

        height = w.data.shape[0]
        for _ in range(self.niters):
            v = self._normalize(torch.mv(torch.t(w.view(height, -1).data), u))
            u = self._normalize(torch.mv(w.view(height, -1).data, v))

        setattr(self.hull, self.pname + "_u", u)
        w.data = w.data / torch.dot(u, torch.mv(w.view(height, -1).data, v))

    def _register(self):
        w = getattr(self.hull, self.pname)
        height = w.data.shape[0]
        u = self._normalize(w.data.new(height).normal_(0, 1))

        self.hull.register_buffer(self.pname + "_u", u)

    def run(self, *args, **kwargs):
        self._updateUV()
        return self.craft.run(*args, **kwargs)
    '''

    def run(self, *args, **kwargs):
        return self.craft.run(*args, **kwargs)



# ------------------------------------ Loss ------------------------------------ #

class MAE(Craft):
    def __init__(self, scope='MAE'):
        super(MAE, self).__init__(scope)
        self.cost = nn.L1Loss()

    def run(self, x, y):
        z = self.cost(x, y)
        return z



class MSE(Craft):
    def __init__(self, scope='MSE'):
        super(MSE, self).__init__(scope)
        self.cost = nn.MSELoss()

    def run(self, x, y):
        z = self.cost(x, y)
        return z



class Huber(Craft):
    def __init__(self, delta=1., scope='HUBER'):
        #|> set l = |x-y|
        #   SmoothL1  = l**2 / 2               , when l < 1
        #   SmoothL1  = l - 1/2                , otherwise
        #   Huber     = l**2 / 2               , when l < delta
        #   Huber     = l*delta - (delta**2)/2 , otherwise
        #|> set l' = l/delta
        #   SmoothL1' = l**2 / (2 * delta**2) = Huber / delta**2 , when l < delta
        #   SmoothL1' = l/delta - 1/2         = Huber / delta**2 , otherwise
        super(Huber, self).__init__(scope)
        if PT_VER >= ver2num('1.9.0'):
            self._fit_closely = False
            self.cost = nn.HuberLoss(delta=delta)
        else:
            self._fit_closely = True
            self.delta = delta
            self.cost = nn.SmoothL1Loss()

    def run(self, x, y):
        if self._fit_closely:
            z = self.delta * self.delta * self.cost(x/self.delta, y/self.delta)
        else:
            z = self.cost(x, y)
        return z



class Charbon(Craft):
    def __init__(self, eps=1e-6, scope='CHARBON'):
        super(Charbon, self).__init__(scope)
        self.eps = eps

    def run(self, x, y):
        z = torch.mean(torch.sqrt((x - y)**2 + self.eps))
        return z



class SigmXE(Craft):
    def __init__(self, scope='SIGMXE'):
        super(SigmXE, self).__init__(scope)
        self.cost = nn.BCEWithLogitsLoss()

    def run(self, x, y):
        z = self.cost(x, y)
        return z



class SftmXE(Craft):
    def __init__(self, is_one_hot, scope='SFTMXE'):
        super(SftmXE, self).__init__(scope)
        self.ioh = is_one_hot
        self.cost = nn.CrossEntropyLoss()

    def run(self, x, y):
        if self.ioh:
            y = torch.argmax(y, dim=-1)
        z = self.cost(x, y)
        return z



class OHEM(Craft):
    def __init__(self, loss_fn, he_ratio=0.005, np_ratio=3., thr=0.01, chnl_wise=True, keep_pos=False, scope='OHEM'):
        super(OHEM, self).__init__(scope)
        self.loss_fn = loss_fn
        self.he_ratio = he_ratio
        self.np_ratio = np_ratio
        self.thr = thr
        self.keep_pos = keep_pos
        if chnl_wise:
            self.ohem = self._channel_mining
        else:
            self.ohem = self._global_mining

    def _global_mining(self, x, y, m):
        loss = self.loss_fn(x, y)
        tile_shape = len(loss.shape) * [1]
        C, H, W = loss.shape[-3:]
        loss = loss.view(-1, H * W)
        if m is not None:
            tile_shape[-3] = C
            m = m.repeat(*tile_shape)
            m = m.view(-1, H * W)
            loss *= m

        pos_mask = (loss < self.thr).float()
        neg_mask = 1 - pos_mask
        if m is not None:
            pos_mask *= m
            neg_mask *= m
        npos = pos_mask.sum()
        nneg = neg_mask.sum()
        npix = loss.numel()
        if m is not None:
            npix *= m.mean()
        nhe = int(max(self.he_ratio * npix, min(self.np_ratio * npos, nneg)))

        neg_loss = (neg_mask * loss).view(-1)
        he_loss, _ = torch.sort(neg_loss, descending=True)
        he_mask = torch.zeros_like(neg_loss)
        he_mask[:nhe] = torch.ones(nhe, device=neg_loss.device)
        he_loss = (he_mask * he_loss).view(loss.shape)

        if self.keep_pos:
            loss = he_loss + pos_mask * loss
            loss = torch.sum(loss) / (npos + nhe)
        else:
            loss = torch.sum(he_loss) / nhe
        return loss

    def _channel_mining(self, x, y, m):
        loss = self.loss_fn(x, y)
        tile_shape = len(loss.shape) * [1]
        C, H, W = loss.shape[-3:]
        loss = loss.view(-1, H * W)
        if m is not None:
            tile_shape[-3] = C
            m = m.repeat(*tile_shape)
            m = m.view(-1, H * W)
            loss *= m

        pos_mask = (loss < self.thr).float()
        neg_mask = 1 - pos_mask
        if m is not None:
            pos_mask *= m
            neg_mask *= m
        npos = pos_mask.sum(-1)
        nneg = neg_mask.sum(-1)
        nhe = torch.where(nneg > self.np_ratio * npos, self.np_ratio * npos, nneg)
        nhe = torch.clamp(nhe, min=self.he_ratio * loss[0].numel())

        neg_loss = neg_mask * loss
        he_idx = torch.argsort(neg_loss.detach(), descending=True)
        he_mask = torch.zeros_like(neg_loss)
        for i in range(he_mask.size(0)):
            if m is not None:
                nhe[i] *= m[i].mean()
            he_mask[i].scatter_(0, he_idx[i, :ceil(nhe[i])], 1.)
        nhe = torch.ceil(nhe)
        he_loss = he_mask * neg_loss

        if self.keep_pos:
            loss = he_loss + pos_mask * loss
            loss = torch.sum(loss, -1) / (npos + nhe)
        else:
            loss = torch.sum(he_loss, -1) / nhe

        return loss.mean()

    def run(self, x, y, m=None):
        z = self.ohem(x, y, m)
        return z



# ------------------------------------ Metric ------------------------------------ #

class AccCls(Craft):
    def __init__(self, multi_class, is_one_hot, scope='ACCCLS'):
        super(AccCls, self).__init__(scope)
        if multi_class:
            assert not is_one_hot
        self.mulcls = multi_class
        self.ioh = is_one_hot

    def run(self, x, y):
        if self.mulcls: # include binary classification as well
            x = torch.round(x)
            correct = torch.mean((x == y).float(), dim=-1)
            z = torch.mean((correct == 1).float())
        else:
            if self.ioh:
                y = torch.argmax(y, dim=-1)
            x = torch.argmax(x, dim=-1)
            z = torch.mean((x == y).float())
        return z



class PSNR(Craft):
    def __init__(self, peak, scope='PSNR'):
        super(PSNR, self).__init__(scope)
        self.peak = peak

    def run(self, x, y):
        mse = torch.mean((x - y) ** 2) + 1e-8
        psnr = 10 * torch.log10(self.peak ** 2 / mse)
        return psnr



class SSIM(Craft):
    def __init__(self, peak, eps1=1e-4, eps2=9e-4, eps3=4.5e-4, scope='SSIM'):
        super(SSIM, self).__init__(scope)
        self.eps1 = eps1 * peak * peak
        self.eps2 = eps2 * peak * peak
        self.eps3 = eps3 * peak * peak

    def run(self, x, y):
        mu_x = torch.mean(x, (1, 2, 3), keepdim=True)
        mu_y = torch.mean(y, (1, 2, 3), keepdim=True)
        sigma_x = torch.sqrt(torch.mean((x - mu_x) ** 2, (1, 2, 3), keepdim=False))
        sigma_y = torch.sqrt(torch.mean((y - mu_y) ** 2, (1, 2, 3), keepdim=False))
        cov = torch.mean((x - mu_x) * (y - mu_y), (1, 2, 3), keepdim=False)
        mu_x = mu_x.squeeze()
        mu_y = mu_y.squeeze()
        luminance = (2 * mu_x * mu_y + self.eps1) / (mu_x ** 2 + mu_y ** 2 + self.eps1)
        contrast = (2 * sigma_x * sigma_y + self.eps2) / (sigma_x ** 2 + sigma_y ** 2 + self.eps2)
        structure = (cov + self.eps3) / (sigma_x * sigma_y + self.eps3)
        ssim = torch.mean(luminance * contrast * structure)
        return ssim



# ------------------------------------ Optimizer ------------------------------------ #

class WarmUpWrapper():
    def __init__(self, warmup, scheduler):
        if PT_VER >= ver2num('2.0.0'):
            scheduler_base = torch.optim.lr_scheduler.LRScheduler
        else:
            scheduler_base = torch.optim.lr_scheduler._LRScheduler
        self.warmup = warmup
        self.mile = -1
        if issubclass(type(scheduler), scheduler_base):
            self.scheduler = scheduler
            self.optz = scheduler.optimizer
        else:
            self.scheduler = None
            self.optz = scheduler
        self.lr_base = self.optz.defaults['lr']
        self.__lr = []
        for group in self.optz.param_groups:
            group['lr'] = 0.

    @property
    def lr(self):
        return self.__lr

    def step(self):
        self.mile += 1
        if self.mile<=self.warmup:
            self.__lr = []
            lr = self.lr_base * self.mile / self.warmup
            for group in self.optz.param_groups:
                group['lr'] = lr
                self.__lr.append(lr)
        elif self.scheduler is not None:
            self.scheduler.step()
            self.__lr = self.scheduler.get_last_lr()



class StepLR(object):
    def __init__(self, period, factor):
        self.period = period
        self.factor = factor

    def __call__(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, self.period, gamma=self.factor)



class PolyLR(object):
    def __init__(self, cutoff, power):
        self.cutoff = cutoff
        self.power = power

    def __call__(self, optimizer):
        lr = optimizer.defaults['lr']
        lr_update = lambda mile: ((lr - 1e-4) * (1 - min(mile / self.cutoff, 1)) ** self.power) + 1e-4
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_update)



class CosLR(object):
    def __init__(self, period):
        self.period = period

    def __call__(self, optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.period,
                                                          eta_min=optimizer.defaults['lr']*0.001)



class ExpLR(object):
    def __init__(self, period, factor):
        self.period = period
        self.factor = factor

    def __call__(self, optimizer):
        lr = optimizer.defaults['lr']
        lr_update = lambda mile: lr * self.factor ** (mile / self.period)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_update)



class WavyLR(object):
    def __init__(self, period, dampen):
        self.period = period
        self.dampen = dampen

    def __call__(self, optimizer):
        lr_update = lambda x: self.dampen ** (x - 1)
        return torch.optim.lr_scheduler.CyclicLR(optimizer, optimizer.defaults['lr'] / 2, optimizer.defaults['lr'],
                                                 step_size_up=self.period // 2, scale_fn=lr_update, scale_mode='cycle')



try:
    from torch._dynamo import OptimizedModule
    compiled_mod = OptimizedModule
except ImportError:
    compiled_mod = str # whatever


class OptzABC(Craft):
    def __init__(self, hull, lr, lr_decay, warmup, mixp,
                grad_limit, grad_accum, update_scope, scope):
        super(OptzABC, self).__init__(scope)
        self.lr_base = lr
        self.__lr = []
        self.mode = 0
        if warmup>0:
            self.mode = 1
        elif lr_decay is not None:
            self.mode = 2

        if PT_VER >= ver2num('2.0.0') and isinstance(hull, compiled_mod):
            hull = hull._orig_mod
        # select parameters await updating
        if update_scope is None:
            update_var = getattr(hull, 'vars', hull.parameters)()
        else:
            if isinstance(update_scope, str):
                update_scope = [update_scope]
            update_var = []
            for us in update_scope:
                paths = us.split('/')
                craft = hull
                for p in paths[1:]:
                    craft = getattr(craft, p)
                update_var.append(getattr(craft, 'vars', craft.parameters)())
        self.update_var = update_var
        self.grad_accum = grad_accum
        self.grad_limit = grad_limit
        self.mixp = mixp
        if mixp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.cnt = 0

    @property
    def lr(self):
        if self.mode == 0:
            return self.lr_base
        elif self.mode == 1:
            self.__lr = self.lr_decay.lr
        else:
            self.__lr = self.lr_decay.get_last_lr()

        if len(self.__lr) == 0:
            return self.lr_base
        elif len(self.__lr) == 1:
            return self.__lr[0]
        else:
            return self.__lr

    def run(self, target):
        self.optz.zero_grad()
        if self.mixp:
            self.scaler.scale(target).backward()
            self.cnt += 1
            if self.cnt == self.grad_accum:
                self.cnt = 0
                if self.grad_limit > 0:
                    self.scaler.unscale_(self.optz)
                    nn.utils.clip_grad_value_(self.update_var, self.grad_limit)
                self.scaler.step(self.optz)
                self.scaler.update()
                if self.lr_decay is not None:
                    self.lr_decay.step()
        else:
            target.backward()
            self.cnt += 1
            if self.cnt == self.grad_accum:
                self.cnt = 0
                if self.grad_limit > 0:
                    nn.utils.clip_grad_value_(self.update_var, self.grad_limit)
                self.optz.step()
                if self.lr_decay is not None:
                    self.lr_decay.step()



class Momentum(OptzABC):
    def __init__(self, hull, lr, mmnt=0.9, wd=0, lr_decay=None, warmup=0, mixp=False,
                 grad_limit=-1, grad_accum=1, update_scope=None, scope='MOMENTUM'):
        super(Momentum, self).__init__(hull, lr, lr_decay, warmup, mixp,
                                grad_limit, grad_accum, update_scope, scope)
        self.optz = torch.optim.SGD(self.update_var, lr=lr, momentum=mmnt, weight_decay=wd)
        if lr_decay is None:
            self.lr_decay = None
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.optz)
        else:
            self.lr_decay = lr_decay(self.optz)
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.lr_decay)



class Nesterov(OptzABC):
    def __init__(self, hull, lr, mmnt=0.9, wd=0, lr_decay=None, warmup=0, mixp=False,
                 grad_limit=-1, grad_accum=1, update_scope=None, scope='NESTEROV'):
        super(Nesterov, self).__init__(hull, lr, lr_decay, warmup, mixp,
                                grad_limit, grad_accum, update_scope, scope)
        self.optz = torch.optim.SGD(self.update_var, lr=lr, momentum=mmnt, weight_decay=wd, nesterov=True)
        if lr_decay is None:
            self.lr_decay = None
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.optz)
        else:
            self.lr_decay = lr_decay(self.optz)
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.lr_decay)



class RMSProp(OptzABC):
    def __init__(self, hull, lr, mmnt=0., wd=0, lr_decay=None, warmup=0, mixp=False,
                 grad_limit=-1, grad_accum=1, update_scope=None, scope='RMSPROP'):
        super(RMSProp, self).__init__(hull, lr, lr_decay, warmup, mixp,
                                grad_limit, grad_accum, update_scope, scope)
        self.optz = torch.optim.RMSprop(self.update_var, lr=lr, momentum=mmnt, weight_decay=wd)
        if lr_decay is None:
            self.lr_decay = None
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.optz)
        else:
            self.lr_decay = lr_decay(self.optz)
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.lr_decay)



class Adam(OptzABC):
    def __init__(self, hull, lr, mmnt1=0.9, mmnt2=0.999, wd=0, lr_decay=None, warmup=0,
                 mixp=False, grad_limit=-1, grad_accum=1, update_scope=None, scope='ADAM'):
        super(Adam, self).__init__(hull, lr, lr_decay, warmup, mixp,
                                grad_limit, grad_accum, update_scope, scope)
        self.optz = torch.optim.Adam(self.update_var, lr=lr, betas=(mmnt1, mmnt2), weight_decay=wd)
        if lr_decay is None:
            self.lr_decay = None
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.optz)
        else:
            self.lr_decay = lr_decay(self.optz)
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.lr_decay)



class AdamW(OptzABC):
    def __init__(self, hull, lr, mmnt1=0.9, mmnt2=0.999, wd=0, lr_decay=None, warmup=0,
                 mixp=False, grad_limit=-1, grad_accum=1, update_scope=None, scope='ADAMW'):
        super(AdamW, self).__init__(hull, lr, lr_decay, warmup, mixp,
                                grad_limit, grad_accum, update_scope, scope)
        self.optz = torch.optim.AdamW(self.update_var, lr=lr, betas=(mmnt1, mmnt2), weight_decay=wd)
        if lr_decay is None:
            self.lr_decay = None
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.optz)
        else:
            self.lr_decay = lr_decay(self.optz)
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.lr_decay)



class _Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.
        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class Lion(OptzABC):
    def __init__(self, hull, lr, mmnt1=0.9, mmnt2=0.99, wd=0, lr_decay=None, warmup=0,
                 mixp=False, grad_limit=-1, grad_accum=1, update_scope=None, scope='LION'):
        super(Lion, self).__init__(hull, lr, lr_decay, warmup, mixp,
                                grad_limit, grad_accum, update_scope, scope)
        self.optz = _Lion(self.update_var, lr=lr, betas=(mmnt1, mmnt2), weight_decay=wd)
        if lr_decay is None:
            self.lr_decay = None
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.optz)
        else:
            self.lr_decay = lr_decay(self.optz)
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.lr_decay)



class _Lamb(torch.optim.Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(_Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss


class Lamb(OptzABC):
    def __init__(self, hull, lr, mmnt1=0.9, mmnt2=0.999, wd=0, lr_decay=None, warmup=0,
                 mixp=False, grad_limit=-1, grad_accum=1, update_scope=None, scope='LION'):
        super(Lamb, self).__init__(hull, lr, lr_decay, warmup, mixp,
                                grad_limit, grad_accum, update_scope, scope)
        self.optz = _Lamb(self.update_var, lr=lr, betas=(mmnt1, mmnt2), weight_decay=wd)
        if lr_decay is None:
            self.lr_decay = None
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.optz)
        else:
            self.lr_decay = lr_decay(self.optz)
            if warmup>0:
                self.lr_decay = WarmUpWrapper(warmup, self.lr_decay)