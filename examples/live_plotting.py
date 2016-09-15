# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:55:21 2015

@author: tud205521
"""

#%%
import time
import qtt
import qtt.legacy
from qtt import pmatlab
import numpy as np
import pyqtgraph as pg
import qtpy.QtWidgets as QtWidgets
import qtpy.QtCore as QtCore
import logging

#%% Liveplot object


class livePlot:
    """ Class to enable live plotting of data """

    def __init__(self, gates, sweepgates, sweepranges, verbose=1):
        win = pg.GraphicsWindow(title="Live view")
        win.resize(800, 600)
        win.move(-900, 10)

        # TODO: automatic scaling?
        # TODO: implement FPGA callback in qcodes
        # TODO: implement 2 function plot (for 2 sensing dots)

        self.win = win
        self.verbose = verbose
        self.idx = 0
        self.maxidx = 1e9
        self.data = None
        self.sweepranges = sweepranges
        self.gates = gates
        self.sweepgates = sweepgates
        self.fps = pmatlab.fps_t(nn=6)

        if len(sweepgates) == 1:
            p1 = win.addPlot(title="Sweep")
            p1.setLabel('left', 'Value')
            p1.setLabel('bottom', self.sweepgates[0], units='mV')
            dd = np.zeros((0,))
            plot = p1.plot(dd, pen='b')
            self.plot = plot
        elif len(sweepgates) == 2:
            p1 = win.addPlot(title='2d scan')
            p1.setLabel('bottom', sweepgates[0], units='mV')
            p1.setLabel('left', sweepgates[1], units='mV')
            self.img = pg.ImageItem()
            p1.addItem(self.img)
        else:
            raise Exception(
                'The number of sweepgates should be either 1 or 2.')

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updatebg)

    def resetdata(self):
        self.idx = 0
        self.data = None

    def update(self, data=None, processevents=True):
        self.win.setWindowTitle('live view, fps: %.2f' % self.fps.framerate())
        if self.verbose >= 2:
            print('livePlot: update')
        if len(self.sweepgates) == 1:
            if data is not None:
                self.data = np.array(data)
                gateval = self.gates.get(self.sweepgates[0])
                sweepvalues =  np.arange(gateval - self.sweepranges[0] / 2 , self.sweepranges[0] / 2 + gateval, self.sweepranges[0] / len(data))
                self.plot.setData(sweepvalues, self.data)
            else:
                pass
        elif len(self.sweepgates) == 2:
            if data is not None:
                self.img.setImage(data.T)
                self.horz_low = self.gates.get(
                    self.sweepgates[0]) - self.sweepranges[0] / 2
                self.horz_range = self.sweepranges[0]
                self.vert_low = self.gates.get(
                    self.sweepgates[1]) - self.sweepranges[1] / 2
                self.vert_range = self.sweepranges[1]
                self.rect = QtCore.QRect(
                    self.horz_low, self.vert_low, self.horz_range, self.vert_range)
                self.img.setRect(self.rect)
        else:
            if self.data is None:
                self.data = np.array(data).reshape((1, -1))
            else:
                self.data = np.vstack((self.data, data))
            self.img.setImage(self.data.T)

        self.idx = self.idx + 1
        if self.idx > self.maxidx:
            self.idx = 0
            self.timer.stop()
        if processevents:
            QtWidgets.QApplication.processEvents()
        pass

    def updatebg(self):
        if self.idx % 10 == 0:
            logging.debug('livePlot: updatebg %d' % self.idx)
        self.idx = self.idx + 1
        self.fps.addtime(time.time())
        if self.datafunction is not None:
            try:
                dd = self.datafunction()
                self.update(data=dd)
            except Exception as e:
                print(e)
                self.stopreadout()
        else:
            self.stopreadout()
            dd = None

    def startreadout(self, callback=None, rate=10., maxidx=None):
        if maxidx is not None:
            self.maxidx = maxidx
        if callback is not None:
            self.datafunction = callback
        self.timer.start(1000 * (1. / rate))
        if self.verbose:
            print('live_plotting: start readout: rate %.1f Hz' % rate)

    def stopreadout(self):
        if self.verbose:
            print('live_plotting: stop readout')
        self.timer.stop()

#%% Some default callbacks


class MockCallback_2d:

    def __init__(self, npoints=6400):
        self.npoints = npoints

    def __call__(self):
        data = np.random.rand(self.npoints)
        data_reshaped = data.reshape(80, 80)
        return data_reshaped


class fpgaCallback_sd:

    def __init__(self, station, Naverage=4, fpga_ch=1):
        self.fpga = station.fpga
        self.Naverage = Naverage
        self.FPGA_mode = 0
        self.fpga_ch = fpga_ch
        self.ReadDevice = 'FPGA_ch%d' % self.fpga_ch

    def __call__(self, verbose=0):
        """ Callback function to read a single line of data from the FPGA """
        totalpoints, DataRead_ch1, DataRead_ch2 = self.fpga.readFPGA(
            Naverage=self.Naverage, ReadDevice=[self.ReadDevice])

        if self.fpga_ch == 1:
            datr = DataRead_ch1[1:-1]
        elif self.fpga_ch == 2:
            datr = DataRead_ch2[1:-1]
        else:
            raise Exception('FPGA channel not well specified')

        datr = [x / self.Naverage for x in datr]
#        datr_ch1[:]=[x*mirrorfactor for x in datr_ch1] # minus sign?
        datr_half = datr[:len(datr) // 2 + 1]

        return np.array(datr_half)

import scipy.ndimage as ndimage
import qtt.algorithms.generic


class fpgaCallback_hc:

    def __init__(self, station, ReadDevice, Naverage, resolution, waittime=0):
        self.fpga = station.fpga
        self.Naverage = Naverage
        self.resolution = resolution
        self.ReadDevice = ReadDevice
        self.waittime = waittime
        self.diffsigma = 1
        self.diff = True
        self.diffaxis = 1
        self.smoothing = False
        self.laplace = False

    def __call__(self):
        totalpoints, DataRead_ch1, DataRead_ch2 = self.fpga.readFPGA(
            ReadDevice=self.ReadDevice, Naverage=self.Naverage, waittime=self.waittime)

        if 'FPGA_ch1' in self.ReadDevice:
            datr = DataRead_ch1
        elif 'FPGA_ch2' in self.ReadDevice:
            datr = DataRead_ch2
        else:
            raise Exception('FPGA channel not well specified')

        chunks_ch1 = [datr[x:x + self.resolution[0]]
                      for x in range(0, len(datr), self.resolution[0])]
        chunks_ch1 = [chunks_ch1[i][1:-7] for i in range(0, len(chunks_ch1))]
        Z = chunks_ch1[:-10]

        if self.diff:
            order = 1
        else:
            order = 0

        im_diff = ndimage.gaussian_filter1d(
            Z, axis=self.diffaxis, sigma=self.diffsigma, order=order, mode='nearest')

        if self.smoothing:
            im_diff = qtt.algorithms.generic.smoothImage(im_diff)

        if self.laplace:
            im_diff = ndimage.filters.laplace(im_diff, mode='nearest')

        return np.array(im_diff)
