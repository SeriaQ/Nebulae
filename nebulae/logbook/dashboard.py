#!/usr/bin/env python
'''
dash_board
Created by Seria at 29/12/2018 3:29 PM
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
from ..kit.utility import curve2str
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import pandas as pd
import os
from sys import stdout
from collections.abc import Iterable
from ..rule import ENV_RANK



class DashBoard(object):
    palette = ['#F08080', '#00BFFF', '#FFFF00', '#2E8B57', '#6A5ACD', '#FFD700', '#808080']
    linestyle = ['-', '--', '-.', ':']
    def __init__(self, log_dir='./logbook', window=1, divisor=12, span=60, format=None):
        '''
        :param window: the window length of moving average
        :param format: a list of which the element is format and mode, e.g. ['3f', 'raw']
        '''
        if isinstance(log_dir, str):
            self.log_dir = log_dir
        elif isinstance(log_dir, Iterable):
            self.log_dir = '.'
            self.iterable = log_dir.__iter__()
            self.length = len(log_dir)
            self.cnt = 0
        else:
            'NEBULAE ERROR ⨷ log_dir must be either a path string or an iterator.'
        self.rank = int(os.environ.get(ENV_RANK, -1))
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.window = window
        self.divisor = divisor
        self.span = span
        self.format = format

        self.max_epoch = 0
        self.first_call = True # if it is the first call for self.__call__()
        self.prev_time = -1
        self.win_mile = {}
        self.gauge_mile = {}
        self.gauge_epoch = {}
        self.trail_mile = {}
        self.trail_epoch = {}
        self.is_global = None
        self.is_elastic = None
        self.ptr = 0

    def _getOridinal(self, number):
        remainder = number % 10
        if remainder == 1:
            ordinal = 'st'
        elif remainder == 2:
            ordinal = 'nd'
        elif remainder == 3:
            ordinal = 'rd'
        else:
            ordinal = 'th'
        return ordinal

    def _formatAsStr(self, stage, abbr, value, epoch, mile, mpe):
        form, mode = self.format[abbr]
        if isinstance(form, str):
            form = '%-' + form
        if mode == 'raw':
            return (' %%s ➠ \033[1;36m%s\033[0m |' % form) % (abbr, value)
        elif mode == 'percent':
            return (' %%s ➠ \033[1;36m%s%%%%\033[0m |' % form) % (abbr, value*100)
        elif mode == 'inviz':
            form(stage, epoch, mile, mpe, value)
            return ''
        elif mode == 'tailor':
            string = form(value)
            return ' %s ➠ \033[1;36m%s\033[0m |' % (abbr, string)
        else:
            raise KeyError('%s is an illegal format option.' % mode)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt == self.length:
            self.cnt = 0
        else:
            self({}, self.cnt, 0, self.length, 'ITER', plot=False)
            self.cnt += 1
        return self.iterable.__next__()
    
    def __call__(self, entry, mile, epoch, mpe, stage, interval=1, duration=None, plot=True, wipe=False, flush=0,
                is_global=True, is_elastic=False, in_loop=(-1,), last_for=1):
        if self.rank>0:
            return
        assert not (plot and wipe), 'NEBULAE ERROR ⨷ plot and wipe are mutually exclusive.'
        epoch += 1
        mile += 1
        string_mile = ''
        flag_display = False
        flag_epoch_end = False
        len_loop = len(in_loop)
        assert in_loop[0]<0 or len_loop>1, 'NEBULAE ERROR ⨷ the dashboard should loop through more than one curve.'
        if mile % interval == 0:
            flag_display = True
        if mile % mpe == 0:
            flag_epoch_end = True
            if epoch > self.max_epoch:
                self.max_epoch = epoch
            string_epoch = ''
            cnt = 0
        if self.first_call:
            self.time = time.time()
        items = []
        for abbr, value in entry.items():
            # read gauge every mile
            global_mile = ((epoch-1)*mpe+mile)
            name = stage + ":" + abbr
            items.append(name)
            if flag_display or self.format[abbr][1] == 'inviz':
                string_mile += self._formatAsStr(stage, abbr, value, epoch, mile, mpe)
            if self.format[abbr][1] in ('inviz', 'tailor'):
                if flag_epoch_end:
                    _ = self._formatAsStr(stage, abbr, value, epoch, -1, mpe)
                continue
            if name not in self.win_mile.keys():
                self.win_mile[name] = np.zeros((self.window,))
                self.gauge_mile[name] = []
                self.gauge_epoch[name] = []
            if stage not in self.trail_mile.keys():
                self.trail_mile[stage] = []
                self.trail_epoch[stage] = []
            if value.dtype == np.float16:
                value = value.astype(np.float32)
            self.win_mile[name][(global_mile - 1) % self.window] = value
            if mile == 1: # the start of an epoch
                self.gauge_epoch[name].append(value)
            else:
                self.gauge_epoch[name][-1] += value # accumulate values
            if global_mile < self.window:
                gauge = np.array(self.win_mile[name][:global_mile]).mean()
            else:
                gauge = np.array(self.win_mile[name]).mean()
            self.gauge_mile[name].append(gauge)
            if len(self.trail_mile[stage]) < len(self.gauge_mile[name]): # assume all metric share the same trail
                self.trail_mile[stage].append(global_mile)
            if flag_epoch_end:
                # read gauge every epoch
                self.gauge_epoch[name][-1] /= mpe
                if len(self.trail_epoch[stage]) < len(self.gauge_epoch[name]):  # assume all metric share the same trail
                    self.trail_epoch[stage].append(epoch)
                indicator = self._formatAsStr(stage, abbr, self.gauge_epoch[name][-1], epoch, -1, mpe)
                string_epoch += indicator
                if indicator != '':
                    cnt += 1
        try:
            w = os.get_terminal_size().columns - 1
            if self.first_call:
                self.first_call = False
        except OSError:
            w = 46 + 12*len(entry)
            if self.first_call:
                print('NEBULAE WARNING ◘ terminal size is unknown which may cause a broken graph.')
                self.first_call = False

        if flag_display:
            if len(self.gauge_mile) > 0 and len(entry)>0 and plot:
                curve_exists = True
            else:
                curve_exists = False
            if curve_exists:
                if len_loop > 1:
                    if mile % (interval*last_for) == 0:
                        self.ptr = (self.ptr + 1) % len_loop
                    metric_idx = in_loop[self.ptr]
                else:
                    metric_idx = 0
                data = np.array(self.gauge_mile[items[metric_idx]])
                is_global = is_global if self.is_global is None else self.is_global
                is_elastic = is_elastic if self.is_elastic is None else self.is_elastic
                curve = curve2str(data, self.divisor, self.span, is_global, is_elastic,
                                  x_title='step', y_title=items[metric_idx] + 10*' ')
                print(curve)
                print(w * ' ', end='\r')

            ordinal = self._getOridinal(epoch)
            progress = int((mile - 1) / mpe * 20 + 0.4)
            yellow_bar = progress * ' '
            space_bar = (20 - progress) * ' '
            if duration is None:
                if self.prev_time < 0:
                    duration = '--:--'
                    self.prev_time = time.time()
                else:
                    curr_time = time.time()
                    duration = '%.3f'%(curr_time - self.prev_time)
                    self.prev_time = curr_time
            else:
                duration = '%.3f'%duration
            if wipe:
                end_char = '\r'
            else:
                end_char = '\n'
            print('| %d%s Epoch ✇ %d Miles ⊰⟦\033[43m%s\033[0m%s⟧⊱︎ ⧲ %ss/mile | %s |%s     '
                  % (epoch, ordinal, mile, yellow_bar, space_bar, duration, stage, string_mile), end=end_char)
            if wipe:
                stdout.flush()
            else:
                if curve_exists:
                    print(f'\033[{self.divisor+flush+7}A')
                else:
                    print(f'\033[2A')
            
        if flag_epoch_end:
            if wipe:
                print(2 * w * ' ', end='\r')
                stdout.flush()
            else:
                if plot:
                    vertical = self.divisor
                else:
                    vertical = -2
                for _ in range(vertical+flush+6):
                    print(w * ' ')
                print(f'\033[{vertical+flush+7}A')
            ordinal = self._getOridinal(epoch)
            mileage = str(epoch * mpe)
            display = '| %d%s Epoch ✇ %s Miles ︎⧲ %.2fs/epoch | %s |%s' \
                      % (epoch, ordinal, mileage, time.time() - self.time, stage, string_epoch)
            print('+' + (len(display) - 3 - cnt * 11) * '-' + '+' + 30 * ' ')
            print(display)
            print('+' + (len(display) - 3 - cnt * 11) * '-' + '+' + 30 * ' ')
            self.time = time.time()

    def read(self, entry, stage, epoch=-1):
        if self.rank>0:
            return 0
        assert epoch!=0, 'NEBULAE ERROR ⨷ epoch starts from 1.'
        epoch = epoch-1 if epoch>0 else epoch
        return self.gauge_epoch[stage + ':' + entry][epoch]

    def record(self, entry, stage, value):
        if self.rank>0:
            return
        name = stage + ':' + entry
        if name in self.gauge_epoch.keys():
            raise KeyError('NEBULAE ERROR ⨷ %s has been taken in dashboard.'%name)
        else:
            self.gauge_epoch[name] = value

    def _argm(self, arr):
        m = [arr[0], arr[0]] # min, max
        idx = [0, 0]
        for i, a in enumerate(arr):
            if a < m[0]:
                m[0] = a
                idx[0] = i
            if a > m[1]:
                m[1] = a
                idx[1] = i
        idx[0] += 1
        idx[1] += 1
        return m + idx

    def log(self, gauge=True, tachograph=True, history='', subdir=''):
        if self.rank>0:
            return
        if subdir:
            log_dir = os.path.join(self.log_dir, subdir)
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = self.log_dir
        if history:
            for f in os.listdir(history):
                if not f.endswith('csv'):
                    continue
                df = pd.read_csv(os.path.join(history, f), header=0)
                value = df.values
                unit = df.columns[0]
                key = df.columns[1]
                if unit == 'mile':
                    self.trail_mile[key] = (value[:, 0] - value[-1, 0]).tolist() + self.trail_mile[key]
                    self.gauge_mile[key] = value[:, 1].tolist() + self.gauge_mile[key]
                elif unit == 'epoch':
                    self.trail_epoch[key] = (value[:, 0] - value[-1, 0]).tolist() + self.trail_epoch[key]
                    self.gauge_epoch[key] = value[:, 1].tolist() + self.gauge_epoch[key]
                else:
                    raise ValueError('NEBULAE ERROR ⨷ header is either to be "mile" or "epoch", but got %s.' % unit)
        if gauge:
            boards = {}
            # group by metrics
            for k, v in self.format.items():
                if not v[1] in ('inviz', 'tailor'):
                    boards[k] = []
            for k in self.gauge_mile.keys():
                boards[k.split(':')[-1]].append(k)
            # plot
            sns.set_theme()
            for k in boards.keys():
                for i, b in enumerate(boards[k]):
                    stage, entry = b.split(':')
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ymin, ymax, xmin, xmax = self._argm(self.gauge_mile[b])
                    xoff = len(self.trail_mile[stage]) * 0.02
                    yoff = (ymax - ymin) * 0.02
                    ax.annotate(('%%%s' % (self.format[entry][0])) % ymin, (xmin - xoff, ymin + yoff))
                    ax.annotate(('%%%s' % (self.format[entry][0])) % ymax, (xmax - xoff, ymax + yoff))
                    data = pd.DataFrame({b: self.gauge_mile[b]}, index=self.trail_mile[stage])
                    sns.lineplot(data=data, markers=False, ax=ax)
                    plt.savefig(os.path.join(log_dir, '%s_%s_%.3g_mile_%d.jpg'
                                    % (k.replace('/', '-'), stage, self.gauge_mile[b][-1], self.trail_mile[stage][-1])))
                    plt.close()

            for k in boards.keys():
                fig = plt.figure()
                ax = fig.add_subplot(111)
                stage = 'UNK'
                for i, b in enumerate(boards[k]):
                    if len(self.gauge_epoch[b])>0:
                        stage, entry = b.split(':')
                        ymin, ymax, xmin, xmax = self._argm(self.gauge_epoch[b])
                        xoff = len(self.trail_epoch[stage]) * 0.02
                        yoff = (ymax - ymin) * 0.02
                        ax.annotate(('%%%s' % (self.format[entry][0])) % ymin, (xmin - xoff, ymin + yoff))
                        ax.annotate(('%%%s' % (self.format[entry][0])) % ymax, (xmax - xoff, ymax + yoff))
                if len(boards[k]) > 0 and self.max_epoch > 0:
                    data = pd.DataFrame({b: np.asarray(self.gauge_epoch[b]) for b in boards[k]},
                                        index=self.trail_epoch[stage])
                    sns.lineplot(data=data, markers=True, ax=ax)
                    plt.savefig(os.path.join(log_dir, '%s_epoch_%d.jpg' % (k, self.max_epoch)))
                plt.close()
        if tachograph:
            for k in self.gauge_mile.keys():
                stage, metric = k.split(':')
                df = pd.DataFrame(data={'mile': self.trail_mile[stage], k: self.gauge_mile[k]})
                df.to_csv(os.path.join(log_dir, '%s_%s_mile.csv'%(metric.replace('/', '-'), stage)), index=None)
                df = pd.DataFrame(data={'epoch': self.trail_epoch[stage], k: self.gauge_epoch[k]})
                df.to_csv(os.path.join(log_dir, '%s_%s_epoch.csv' % (metric.replace('/', '-'), stage)), index=None)


if __name__ == "__main__":
    mode = 'iter'
    if mode == 'iter':
        from time import sleep
        for i in DashBoard(range(10)):
            sleep(1)
    elif mode == 'log':
        import random as rand
        db = DashBoard(log_dir="/Users/Seria/Desktop/nebulae/test/ckpt",
                        window=15, divisor=15, span=70,
                        format={"Acc": [".2f", "percent"], "Loss": [".3f", "raw"], "MAE": [".3f", "raw"]})
        for epoch in range(5):
            for mile in range(10):
                probe = {'Acc':rand.random(), 'Loss':rand.random()}
                db(probe, mile, epoch, 10, 'TRAIN', interval=1, is_global=True)
        
            for mile in range(10):
                probe = {'Acc':rand.random()}
                db(probe, mile, epoch, 10, 'DEV', interval=1, is_global=True)
        db.log()