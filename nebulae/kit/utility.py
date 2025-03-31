#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import yaml
import numpy as np
from scipy import sparse
import subprocess as subps
import h5py
import os
import io
from math import ceil
from time import sleep
from PIL import Image
import cv2
from ..rule import ENV_RANK, FRAME_KEY, CHAR_SEP



def autopad(kernel: tuple, stride=1, dilation=1, size=None):
    '''
    Args
    - in_size: input size e.g. (h, w) for 2d tensor.
               it is None only if input size can be divided by stride along all dimensions
    - kernel: kernel size
    - stride: convolution stride
    - dilation: stride in atrous convolution

    Return
    padding elements on dimensions in reverse order
    e.g. (left, right, top, bottom, front, back) for 3d tensor
    '''
    dim = len(kernel)
    if isinstance(stride, int):
        stride = dim * [stride]
    if isinstance(dilation, int):
        dilation = dim * [dilation]

    padding = []
    homo_padding = []
    if size is None:
        for d in range(dim - 1, -1, -1):
            margin = - stride[d] + kernel[d] + (dilation[d] - 1) * (kernel[d] - 1)
            padding.extend([margin // 2, margin - margin // 2])
            if margin%2 == 0:
                homo_padding.append(margin//2)
    else:
        for d in range(dim-1,-1,-1):
            margin = (ceil(size[d] / stride[d]) - 1) * stride[d] + kernel[d] + \
                     (dilation[d] - 1) * (kernel[d] - 1) - size[d]
            padding.extend([margin//2, margin-margin//2])
            if margin%2 == 0:
                homo_padding.append(margin//2)

    if len(homo_padding) == dim:
        return homo_padding
    else:
        return padding

def cap(key, scope):
    return scope + '/' + key

def doff(key):
    return key.split('/')[-1]

def ver2num(version, vbits=2):
    version = version.split('.')
    number = 0
    for v in version:
        v = v.split('+')
        if len(v)>1:
            v = int(v[0], 16)
        else:
            v = int(v[0])
        number = number * 10 ** vbits + v
    return number


def sprawl(root, desc=None, body='', branch=(), depth=-1, layer=0, outfile:object=None):
    if isinstance(root, str) and os.path.isdir(root):
        if layer == 0:
            line = '|⊻| ' + os.path.basename(root)
            if outfile is not None:
                outfile.write(line + '\n')
            else:
                print(line)
        traverse = os.listdir(root)
        for i, f in enumerate(traverse):
            if i < len(traverse) - 1:
                line = (layer + 1) * 2 * ' ' + '├─' + f
                if outfile is not None:
                    outfile.write(line + '\n')
                else:
                    print(line)
            else:
                line = (layer + 1) * 2 * ' ' + '└─' + f
                if outfile is not None:
                    outfile.write(line + '\n')
                else:
                    print(line)
            if os.path.isdir(os.path.join(root, f)) and (depth<0 or layer<depth):
                sprawl(os.path.join(root, f), desc, body, branch, depth, layer + 1, outfile)
    else:
        if layer == 0:
            itself = root if desc is None else getattr(root, desc)
            line = '|⊻| ' + itself + ': ' + ' '.join([f'{getattr(root, b)}' for b in branch])
            if outfile is not None:
                outfile.write(line + '\n')
            else:
                print(line)
        traverse = getattr(root, body)
        for i, f in enumerate(traverse):
            itself = f if desc is None else getattr(f, desc)
            if i < len(traverse) - 1:
                line = (layer + 1) * 2 * ' ' + '├─' + itself + ': ' + ' '.join([f'{b}={getattr(f, b)}' for b in branch])
                if outfile is not None:
                    outfile.write(line + '\n')
                else:
                    print(line)
            else:
                line = (layer + 1) * 2 * ' ' + '└─' + itself + ': ' + ' '.join([f'{b}={getattr(f, b)}' for b in branch])
                if outfile is not None:
                    outfile.write(line + '\n')
                else:
                    print(line)
            if hasattr(root, body) and (depth<0 or layer<depth):
                sprawl(f, desc, body, branch, depth, layer + 1, outfile)
    # close file object
    if outfile is not None and layer == 0:
        outfile.close()


def hotvec2mtx(labels, nclasses, on_value=1, off_value=0):
    '''
    Args
    labels: there are three possible format as follows,
            1. list of str
            e.g. ['0,2', '1', '1,2,3']
            2. list of array
            e.g. [[0,2], [1], [1,2,3]]
            3. list of int
            e.g. [0, 2, 1]
    nclasses: number of classes
    on_value: the value on behalf of positive label
    off_value: the value on behalf of negative label

    Returns
    dense label matrix
    '''
    batch_size = labels.shape[0]
    # initialize dense labels
    dense = off_value * np.ones((batch_size * nclasses), dtype=np.float32)
    indices = []
    if isinstance(labels[0], str):
        for b in range(batch_size):
            indices += [int(s) + b * nclasses for s in labels[b].split(CHAR_SEP)]
    elif isinstance(labels[0], (list, np.ndarray)): # labels is a nested array
        for b in range(batch_size):
            indices += [l + b * nclasses for l in labels[b]]
        dense[indices] = on_value
    else:
        for b in range(batch_size):
            indices += [int(labels[b]) + b * nclasses]
        dense[indices] = on_value
    return np.reshape(dense, (batch_size, nclasses))


def mtx2hotvec(labels):
    hot = []
    row, col = labels.shape
    for y in range(row):
        vec = []
        for x in range(col):
            if labels[y, x] != 0:
                vec.append(x)
        hot.append(vec)
    return hot


def den2spa(arr):
    return sparse.coo_matrix(arr)


def spa2den(*args):
    if len(args)==1:
        return args[0].todense()
    else:
        arr = sparse.coo_matrix((args[2], (args[0], args[1])), shape=args[-1])
        return arr.todense()


def rand_trunc_norm(mean, std, shape, cutoff_sigma=2):
    norm = np.random.normal(0, 1, (4,)+shape).astype(np.float32)
    valid = np.logical_and(norm>-cutoff_sigma, norm<cutoff_sigma)
    indices = np.argmax(valid, 0)
    norm = np.choose(indices, norm)
    norm *= std
    norm += mean
    return norm


def parse_cfg(config_path):
    if config_path.endswith('yml'):
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
    elif config_path.endswith('json'):
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
    return config

def record_cfg(config_path, config, overwrite=True):
    if config_path.endswith('yml'):
        while not overwrite and os.path.exists(config_path):
            config_path = config_path[:-5]+'_.yml'
        with open(config_path, 'w') as config_file:
            config_file.write(yaml.dump(config, allow_unicode=True, sort_keys=False))
    elif config_path.endswith('json'):
        while not overwrite and os.path.exists(config_path):
            config_path = config_path[:-5]+'_.json'
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)


def _merge_fuel(src_dir, src, dst, dtype, keep_src=True):
    data = {}
    info_keys = []
    shards = len(src)
    print('+' + (23 + 2 * shards + len(dst)) * '-' + '+')
    # read
    ending_char = '\r'
    for i, f in enumerate(src):
        hdf5 = h5py.File(os.path.join(src_dir, f), 'r')
        if i == 0:
            for key in hdf5.keys():
                info_keys.append(key)
            for key in info_keys:
                if key == FRAME_KEY:
                    data[key] = 0
                else:
                    data[key] = []
        for key in info_keys:
            if key == FRAME_KEY:
                frames = hdf5[key]
                data[key] = frames if frames>data[key] else data[key]
            else:
                data[key].extend(hdf5[key][()].tolist())
        hdf5.close()
        if not keep_src:
            os.remove(os.path.join(src_dir, f))

        progress = i+1
        yellow_bar = progress * '❖-'
        space_bar = (shards - progress) * '◇-'
        if progress == shards:
            ending_char = '\n'
        print('| Merging data  \033[1;35m%s\033[0m  ⊰⟦-%s%s⟧⊱ |'
              % (dst, yellow_bar, space_bar), end=ending_char)
    # write
    hdf5 = h5py.File(os.path.join(src_dir, dst), 'w')
    for key in info_keys:
        if dtype[key].startswith('v'):  # dealing with encoded / varied data
            dt = h5py.special_dtype(vlen=dtype[key])
            hdf5.create_dataset(key, dtype=dt, data=np.array(data[key]))
        elif key == FRAME_KEY:
            hdf5[key] = data[key]
        else:
            hdf5[key] = np.array(data[key]).astype(dtype[key])
    hdf5.close()
    print('+' + (23 + 2 * shards + len(dst)) * '-' + '+')

def _fill_fuel(src, key, data):
    hdf5 = h5py.File(src, 'a')
    if isinstance(data, (int, float, str)):
        data = [data]
    if isinstance(data, list):
        data = np.array(data)
    if data.dtype.kind == 'U': # convert unicode to string
        nbyte = data.dtype.descr[0][1].split('U')[-1]
        data_copy = data.copy()
        data = np.empty(data_copy.shape, dtype='|S'+nbyte)
        for idx, elm in np.ndenumerate(data_copy):
            data[idx] = elm.encode()
    if data.dtype.kind == 'S':
        sdt = h5py.special_dtype(vlen=str)
        hdf5.create_dataset(key, dtype=sdt, data=data)
    else:
        hdf5[key] = data
    hdf5.close()

def _deduct_fuel(src, key):
    hdf5 = h5py.File(src, 'a')
    del hdf5[key]
    hdf5.close()

def byte2arr(data_src, as_np=True):
    data_bytes = data_src.tobytes()
    if as_np:
        data_np = np.frombuffer(data_bytes, dtype='uint8')
        data_np = cv2.imdecode(data_np, -1)
        return data_np
    else:
        data_pil = Image.open(io.BytesIO(data_bytes))
        return data_pil

def rgb2y(data_pil, as_np=True):
    data_yuv = data_pil.convert('YCbCr')
    if as_np:
        data_np = np.ndarray((data_yuv.size[1], data_yuv.size[0], 3), 'u1', data_yuv.tobytes())
        return data_np[:, :, 0]
    else:
        y, _, _ = data_yuv.split()
        return y



def qr_Gram_Schmidt(A):
    """Gram-schmidt Orthogonalization"""
    Q=np.zeros_like(A)
    cnt = 0
    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i]) # minus vector projection
        e = u / np.linalg.norm(u)  # normalization
        Q[:, cnt] = e
        cnt += 1
    R = np.dot(Q.T, A)
    return Q, R

def qr_Givens(A):
    """Givens Rotation"""
    (r, c) = np.shape(A)
    Q = np.identity(r)
    R = np.copy(A)
    (rows, cols) = np.tril_indices(r, -1, c)
    for (row, col) in zip(rows, cols):
        if R[row, col] != 0:  # R[row, col]=0 -> c=1,s=0, R and Q stay unchanged
            r_ = np.hypot(R[col, col], R[row, col])  # d
            c = R[col, col]/r_
            s = -R[row, col]/r_
            G = np.identity(r)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s
            R = np.dot(G, R)  # R=G(n-1,n)*...*G(2n)*...*G(23,1n)*...*G(12)*A
            Q = np.dot(Q, G.T)  # Q=G(n-1,n).T*...*G(2n).T*...*G(23,1n).T*...*G(12).T
    return Q, R

def qr_Householder(A):
    """Householder Reflection"""
    (r, c) = np.shape(A)
    Q = np.identity(r)
    R = np.copy(A)
    for cnt in range(r - 1):
        x = R[cnt:, cnt]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        Q_cnt = np.identity(r)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)
        R = np.dot(Q_cnt, R)  # R=H(n-1)*...*H(2)*H(1)*A
        Q = np.dot(Q, Q_cnt)  # Q=H(n-1)*...*H(2)*H(1)
    return Q, R



class GPUtil():
    def __init__(self):
        self.rank = int(os.environ.get(ENV_RANK, -1))
        self.process = None # the monitor process
        self.stat = 'No statistic for now.'
        self.gpus = [] # GPU names
        with subps.Popen(['nvidia-smi', '--query-gpu=name', '--format=csv'], stdout=subps.PIPE) as p:
            _ = p.stdout.readline()
            for line in p.stdout.readlines():
                self.gpus.append(line.decode('utf-8').strip())

    def _stamp2secs(self, t):
        date, time = t.split(' ')
        h, m, s = time.split(':')
        return 3600 * int(h) + 60 * int(m) + float(s)

    def available(self, ngpu, least_mem=1024):
        with subps.Popen(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv'],
                        stdout=subps.PIPE) as p:
            gpu_id = 0 # the next gpu we are about to check
            id_mem = [] # gpu having avialable memory greater than least requirement
            title = p.stdout.readline().decode('utf-8') # skip this line
            for l in p.stdout.readlines():
                name, total, vacancy = l.decode('utf-8').strip().split(', ')
                total = int(total.split(' ')[0])
                vacancy = int(vacancy.split(' ')[0])
                if vacancy > least_mem:
                    if len(id_mem) < ngpu:
                        id_mem.append((gpu_id, name, total, vacancy))  # (id, info...) of gpu
                    else:
                        id_mem.sort(key=lambda x: x[-1])
                        if vacancy > id_mem[0][-1]:
                            id_mem[0] = (gpu_id, name, total, vacancy)
                gpu_id += 1
        return id_mem

    def monitor(self, sec=5):
        if self.rank > 0:
            return
        assert isinstance(sec, int), 'NEBULAE ERROR ៙ the monitoring interval must be an integer.'
        if sec<5:
            print('NEBULAE WARNING ◘ monitor GPU too often might cause abnormal statistics.')
        self.file = open('./_temp_gpu_stat.csv', 'w')
        # time limit is 12 hours
        self.process = subps.Popen(['timeout', '43200', 'nvidia-smi',
                                    '--query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used',
                                    '--format=csv', '-l', '%d'%sec],
                                   stdout=self.file, stderr=subps.PIPE)

    def status(self):
        if self.rank > 0:
            return
        while self.process.poll() != 0: # if monitor hasn't terminated
            subps.call(['pkill', 'nvidia-smi'])
            sleep(1)
        self.file.close()
        with open('./_temp_gpu_stat.csv', 'r') as f:
            gpu_id = 0
            n = len(self.gpus)
            t_ = ''
            prev_s = [-1 for _ in range(n)]
            prev_t = [-1 for _ in range(n)]
            prev_u = [-1 for _ in range(n)]
            prev_m = [-1 for _ in range(n)]
            curr_s = [-1 for _ in range(n)]
            curr_t = [-1 for _ in range(n)]
            curr_u = [-1 for _ in range(n)]
            curr_m = [-1 for _ in range(n)]
            area_t = [0 for _ in range(n)]
            area_u = [0 for _ in range(n)]
            area_m = [0 for _ in range(n)]
            line = f.readline()
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip().split(', ')
                if len(line)!=4 or len(line[0])!=23 or line[-1][-3:]!='MiB': # skip title line and error lines
                    print('NEBULAE WARNING ◘ GPU log file is truncated due to part of damaged numbers.')
                    break
                minute, second = line[0][-9:].split(':')
                curr_s[gpu_id] = 60 * int(minute) + float(second)
                curr_t[gpu_id] = int(line[1])
                curr_u[gpu_id] = int(line[2][:-2])
                curr_m[gpu_id] = int(line[3][:-4]) / 100 # avoid overfloating
                if t_ == '': # have not gone over all gpus
                    if gpu_id == n-1:
                        t_ = line[0]
                else:
                    duration = curr_s[gpu_id] - prev_s[gpu_id]
                    if duration < 0: # cross hours, e.g. from 1:59:59 pm to 2:00:01 pm
                        duration += 1800
                        # in case of disorder, e.g. previous line is 1:59:59 and current line is 1:59:58
                        duration = duration + 1800 if duration<0 else abs(duration-1800)
                    area_t[gpu_id] += duration * (curr_t[gpu_id] + prev_t[gpu_id]) / 2
                    area_u[gpu_id] += duration * (curr_u[gpu_id] + prev_u[gpu_id]) / 2
                    area_m[gpu_id] += duration * (curr_m[gpu_id] + prev_m[gpu_id]) / 2
                prev_s[gpu_id] = curr_s[gpu_id]
                prev_t[gpu_id] = curr_t[gpu_id]
                prev_u[gpu_id] = curr_u[gpu_id]
                prev_m[gpu_id] = curr_m[gpu_id]
                gpu_id = (gpu_id + 1) % n
                _t = line[0]
        t_ = self._stamp2secs(t_)
        _t = self._stamp2secs(_t)
        t = _t - t_
        t = t + 3600 * 24 if t<0 else t
        
        stat = '+' + 67 * '-' + '+'
        stat += '\n| GPU | ' + 10 * ' ' + 'Name' + 11 * ' ' + ' | T.Celsius | V-Util |   Memory   |\n'
        stat += '+' + 67 * '-' + '+'
        for i, g in enumerate(self.gpus):
            stat += '\n| {:^3d} | {:<25s} |  {:<4.1f}ºC   | {:4.1f}%  | {:<7.1f}MiB |'.format(
                i, g, area_t[i]/t, area_u[i]/t, area_m[i]/t*100)
        stat += '\n+' + 67 * '-' + '+'
        self.stat = stat
        os.remove('./_temp_gpu_stat.csv')
        print(self.stat)

if __name__=='__main__':
    class Node():
        def __init__(self, v, name):
            self.v = v
            self.name = name
            self.kids = []
        def cling(self, k):
            self.kids.append(k)

    nodes = []
    for i in range(1, 6):
        nodes.append(Node(i**2, '#%d'%i))
    nodes[0].cling(nodes[1])
    nodes[0].cling(nodes[3])
    nodes[3].cling(nodes[4])
    root = Node(0, 'R')
    root.cling(nodes[0])
    root.cling(nodes[3])

    sprawl(root, 'name', 'kids', 'v')
