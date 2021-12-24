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
from ..toolkit.utility import curve2str
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import os
import warnings



class DashBoard(object):
    palette = ['#F08080', '#00BFFF', '#FFFF00', '#2E8B57', '#6A5ACD', '#FFD700', '#808080']
    linestyle = ['-', '--', '-.', ':']
    def __init__(self, config=None, log_path='./aerolog', window=1, divisor=10, span=30, format=None):
        '''
        :param config:
        :param window: the window length of moving average
        :param format: a list of which the element is format and mode, e.g. ['3f', 'raw']
        '''
        rank = int(os.environ.get('RANK', -1))
        if config is None:
            self.param = {'log_path': log_path, 'window': window, 'divisor': divisor,
                          'span': span, 'format': format, 'rank': rank}
        else:
            config['window'] = config.get('window', window)
            config['divisor'] = config.get('divisor', divisor)
            config['span'] = config.get('span', span)
            config['rank'] = config.get('rank', rank)
            self.param = config
        assert len(self.param['format'])<8, 'NEBULAE ERROR ⨷ there are at most 7 panels to monitor.'
        if not os.path.exists(self.param['log_path']):
            os.mkdir(self.param['log_path'])
        self.max_epoch = 0
        self.first_call = True # if it is the first call for self.gauge()
        self.win_mile = {}
        self.gauge_mile = {}
        self.gauge_epoch = {}
        self.trail_mile = {}
        self.trail_epoch = {}
        self.is_global = True
        self.panel = 0

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
        form, mode = self.param['format'][abbr]
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

    def gauge(self, entry, mile, epoch, mpe, stage, interval=1, duration=None, flush=0):
        if self.param['rank']>0:
            return
        epoch += 1
        mile += 1
        string_mile = ''
        flag_display = False
        flag_epoch_end = False
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
            self.first_call = False
        items = []
        for abbr, value in entry.items():
            # read gauge every mile
            global_mile = ((epoch-1)*mpe+mile)
            name = stage + ":" + abbr
            items.append(name)
            if flag_display or self.param['format'][abbr][1] == 'inviz':
                string_mile += self._formatAsStr(stage, abbr, value, epoch, mile, mpe)
            if self.param['format'][abbr][1] in ('inviz', 'tailor'):
                if flag_epoch_end:
                    _ = self._formatAsStr(stage, abbr, value, epoch, -1, mpe)
                continue
            if name not in self.win_mile.keys():
                self.win_mile[name] = np.zeros((self.param['window'],))
                self.gauge_mile[name] = []
                self.gauge_epoch[name] = []
                self.trail_mile[name] = []
                self.trail_epoch[name] = []
            self.win_mile[name][(global_mile - 1) % self.param['window']] = value
            if mile == 1: # the start of an epoch
                self.gauge_epoch[name].append(value)
            else:
                self.gauge_epoch[name][-1] += value # accumulate values
            if global_mile < self.param['window']:
                gauge = np.array(self.win_mile[name][:global_mile]).mean()
            else:
                gauge = np.array(self.win_mile[name]).mean()
            self.gauge_mile[name].append(gauge)
            self.trail_mile[name].append(global_mile)
            if flag_epoch_end:
                # read gauge every epoch
                self.gauge_epoch[name][-1] /= mpe
                self.trail_epoch[name].append(epoch)
                indicator = self._formatAsStr(stage, abbr, self.gauge_epoch[name][-1], epoch, -1, mpe)
                string_epoch += indicator
                if indicator != '':
                    cnt += 1
        w = os.get_terminal_size().columns - 1
        if flag_display:
            if len(self.gauge_mile) > 0 and len(entry)>0:
                curve_exists = True
            else:
                curve_exists = False
            if curve_exists:
                data = np.array(self.gauge_mile[items[self.panel]])
                curve = curve2str(data, self.param['divisor'], self.param['span'],
                                  self.is_global, x_title='step', y_title=items[self.panel] + 10*' ')
                print(curve)
                print(w * ' ', end='\r')

            ordinal = self._getOridinal(epoch)
            progress = int((mile - 1) / mpe * 20 + 0.4)
            yellow_bar = progress * ' '
            space_bar = (20 - progress) * ' '
            if duration is None:
                duration = '--:--'
            else:
                duration = '%.3f'%duration
            print('| %d%s Epoch ✇ %d Miles ⊰⟦\033[43m%s\033[0m%s⟧⊱︎ ⧲ %ss/mile | %s |%s     '
                  % (epoch, ordinal, mile, yellow_bar, space_bar, duration, stage, string_mile), end='\n')
            if curve_exists:
                print(f'\033[{self.param["divisor"]+flush+7}A')
            else:
                print(f'\033[2A')
        if flag_epoch_end:
            for _ in range(self.param["divisor"]+flush+6):
                print(w * ' ')
            print(f'\033[{self.param["divisor"]+flush+7}A')
            ordinal = self._getOridinal(epoch)
            mileage = str(epoch * mpe)
            display = '| %d%s Epoch ✇ %s Miles ︎⧲ %.2fs/epoch | %s |%s' \
                      % (epoch, ordinal, mileage, time.time() - self.time, stage, string_epoch)
            print('+' + (len(display) - 3 - cnt * 11) * '-' + '+' + 30 * ' ')
            print(display)
            print('+' + (len(display) - 3 - cnt * 11) * '-' + '+' + 30 * ' ')
            self.time = time.time()

    def read(self, entry, stage, epoch=-1):
        if self.param['rank']>0:
            return
        assert epoch!=0, 'NEBULAE ERROR ⨷ epoch starts from 1.'
        epoch = epoch-1 if epoch>0 else epoch
        return self.gauge_epoch[stage + ':' + entry][epoch-1]

    def record(self, entry, stage, value):
        if self.param['rank']>0:
            return
        name = stage + ':' + entry
        if name in self.gauge_epoch.keys():
            raise KeyError('NEBULAE ERROR ⨷ %s has been taken in dashboard.'%name)
        else:
            self.gauge_epoch[name] = value

    def log(self, gauge=True, tachograph=True, history=''):
        if self.param['rank']>0:
            return
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
            # clustering
            for k, v in self.param['format'].items():
                if not v[1] in ('inviz', 'tailor'):
                    boards[k] = []
            for k in self.gauge_mile.keys():
                boards[k.split(':')[-1]].append(k)
            # plot
            warnings.filterwarnings('ignore', message='[ ]*UserWarning: Creating legend with loc*')
            for k in boards.keys():
                for i, b in enumerate(boards[k]):
                    stage = b.split(':')[0]
                    plt.plot(self.trail_mile[b], self.gauge_mile[b], c=self.palette[i % len(self.palette)], label=b)
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(self.param['log_path'], '%s_%s_%.3g_mile_%d.jpg'
                                          % (k.replace('/', '-'), stage, self.gauge_mile[b][-1], self.trail_mile[b][-1])))
                    plt.close()

                for i, b in enumerate(boards[k]):
                    plt.plot(self.trail_epoch[b], self.gauge_epoch[b], marker='o',
                             c=self.palette[i % 7], linestyle=self.linestyle[i % len(self.linestyle)], label=b)
                if self.max_epoch > 0:
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(self.param['log_path'], '%s_epoch_%d.jpg' % (k.replace('/', '-'), self.max_epoch)))
                    plt.close()
        if tachograph:
            for k in self.gauge_mile.keys():
                stage, metric = k.split(':')
                df = pd.DataFrame(data={'mile': self.trail_mile[k], k: self.gauge_mile[k]})
                df.to_csv(os.path.join(self.param['log_path'], '%s_%s_mile.csv'%(metric.replace('/', '-'), stage)),
                           index=None)
                df = pd.DataFrame(data={'epoch': self.trail_epoch[k], k: self.gauge_epoch[k]})
                df.to_csv(os.path.join(self.param['log_path'], '%s_%s_epoch.csv' % (metric.replace('/', '-'), stage)),
                           index=None)