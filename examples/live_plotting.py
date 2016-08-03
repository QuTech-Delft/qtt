# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:55:21 2015

@author: tud205521
"""

#%%

import os,sys, copy, math
import warnings
import qcodes
import logging

if __name__=='__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except:
        pass

import qtt
import qtt.legacy

import numpy as np
import pyqtgraph as pg
import qtpy.QtWidgets as QtWidgets
import qtpy.QtGui as QtGui
import qtpy.QtCore as QtCore

_qtapp=pg.mkQApp()

import virtualV2
virtualV2.initialize(server_name='test2')
gates=virtualV2.gates
    
#%% Liveplot object
#    
#
class livePlot:
    """ Class to enable live plotting of data """
    def __init__(self, sweepgate='sweepgate', mode='1d', plotrange=[None, None], sweepdata=None):
        win = pg.GraphicsWindow(title="Live view" )
        win.resize(800, 600)
        win.move(-900,10)
        win.setWindowTitle('Live view')
        
        # TODO: automatic scaling?
        # TODO: implement FPGA callback in qcodes    
        # TODO: implement 2 function plot (for 2 sensing dots)
        pg.setConfigOptions(antialias=True)
        self.win = win
        self.mode=mode
        self.verbose=1
        self.idx=0
        self.maxidx=20*50
        self.maxidx=1e9
        self.data=None
        
        self.sweepdata=None
        
        self.plotrange=plotrange
        
        if self.mode=='1d':

            p1=win.addPlot(title="Sweep" )
            p1.setLabel('left', 'Value')
            p1.setLabel('bottom', sweepgate, units='mV')
            dd=np.zeros( (0,))
            h=p1.plot(dd, pen='b')
            self.p1 = p1
            self.h=h
            #self.data=dd
            
            pen =pg.mkPen(color='c', style=QtCore.Qt.DotLine)
            self.sweepdata=sweepdata
            if self.sweepdata is None:
                sweepdata=np.arange(dd.size)
            
            self.highplot = self.p1.plot(sweepdata,  0*sweepdata, pen=pen)
            self.lowplot = self.p1.plot(sweepdata,  0*sweepdata, pen=pen)

            self.setHighLow(self.plotrange)
        
        else:
            p1 = win.addPlot(title='2d')
            self.p1=p1
            p1.invertX()
            p1.invertY()
            # Item for displaying image data
            self.img = pg.ImageItem()
            p1.addItem(self.img)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updatebg)
        
        return
            
    def setHighLow(self, plotrange):
        self.plotrange=plotrange
        if self.plotrange[0] is not None:
            self.highplot.setData(self.sweepdata, self.plotrange[1]+0*self.sweepdata)
            self.lowplot.setData(self.sweepdata, self.plotrange[0]+0*self.sweepdata)
        
    def resetdata(self):
        self.idx=0
        self.data=None
    def update(self, data=None, processevents=True):
        if self.verbose>=2:
            print('livePlot: update' )
        #lp.h
        if self.mode=='1d':
            if data is not None:
                self.data=np.array(data)
            if self.sweepdata is not None:
                if self.verbose>=3:
                    print(' sizes' )
                    print(self.sweepdata.shape)
                    print(self.data.shape)
                if self.sweepdata.size != self.data.size:
                    warnings.warn('live_plotting: error with sizes')
                    self.h.setData(np.arange(self.data.size), self.data)
                else:
                    self.h.setData(self.sweepdata, self.data)
            else:
                self.h.setData(self.data)
            if data is not None:
                pass
        else:
            if self.data is None:
                self.data=np.array(data).reshape( (1,-1))
            else:
                self.data=np.vstack( (self.data, data) )
            #QtGui.QApplication.processEvents()
            self.img.setImage(self.data.T)
        
        self.idx = self.idx + 1
        if self.idx>self.maxidx:
            self.idx=0
            self.timer.stop()
        if processevents:
            QtWidgets.QApplication.processEvents()
        pass
    
    def updatebg(self, verbose=0):
        if self.idx%10==0:
            logging.debug('livePlot: updatebg %d' % self.idx)
        self.idx=self.idx+1
        if self.datafunction is not None:
            try:
                dd=self.datafunction()
                self.update(data=dd)
            except Exception as e:
                print(e)
                dd=None
                self.stopreadout()
                
        else:
            self.stopreadout()
            dd=None
        pass
        return dd
    def startreadout(self, callback=None, rate=10., maxidx=None):
        self.verbose=min(self.verbose, 1)
        if maxidx is not None:
            self.maxidx=maxidx
        if callback is not None:
            self.datafunction = callback
        self.timer.start( 1000*(1./rate) )
        if self.verbose:
            print('live_plotting: start readout: rate %.1f Hz' % rate)
        
    def stopreadout(self):
        if self.verbose:
            print('live_plotting: stop readout')
        self.timer.stop()
        
#%% Some default callbacks
import time
   
class fpgaCallback:
    def __init__(self, Naverage=4, risetime=2e-3, FPGA_mode=0):   
        self.Naverage=Naverage
        self.risetime=risetime
        self.FPGA_mode=FPGA_mode
        
    def __call__(self, verbose=0):
        """ Callback function to read a single line of data from the FPGA """
        t0=time.time()
        totalpoints, DataRead_ch1, DataRead_ch2  = measurementfunctions.readFPGA(FPGA_ave, self.FPGA_mode, Naverage=self.Naverage, risetime=self.risetime, verbose=0)
        dt=time.time()-t0
        if verbose:
            print('liveCallback: dt %.1f [ms] (raw measurement %.1f [ms], risetime %.2f [ms])'  % (dt*1e3, 2 * self.risetime * self.Naverage *1e3, self.risetime*1e3))   
           
        ww=np.array(extractFPGAsweep(DataRead_ch1)) / self.Naverage       
        if verbose:
            print('liveCallback: DataRead_ch1 %d -> %d'  % (len(DataRead_ch1), len(ww)) )   
    
        return np.array(ww)    


gates.L.set(0)
gates.R.set(1)
class dummyCallback:
    def __init__(self, npoints=200):   
        self.npoints=npoints
        self.makeCurve()
    def makeCurve(self, verbose=0):
        v1=gates.L.get()
        v2=gates.R.get()
        if verbose:
            print('param: %f %f' % (v1, v2) )
        npoints=self.npoints
        self.curve=((np.arange(npoints)-npoints/2+v1)/30)**3
        self.curve+=((np.arange(npoints)-npoints/2+v1)/30)**2*v2
    def __call__(self, verbose=0):
        """ Callback function to read a single line of data """
        self.makeCurve()
        ww=self.curve + 10*np.random.rand(self.npoints)
        return np.array(ww)    

#%%

if __name__=='__main__':
    
    
  #  dd=im[0,:]
##    
    _qtapp.closeAllWindows()
    
    lp = livePlot(mode='1d')
    lp.win.setGeometry(10,100,500,400)
    lp.sweepdata=np.arange(200)
    lp.setHighLow([-200,800])
    self=lp
    lp.verbose=2

#%% Start readout

if __name__=='__main__':
    lp.sweepdata=np.arange(200)
    lp.datafunction=dummyCallback(200)
    lp.setHighLow([-20,180])
    lp.updatebg()
    lp.startreadout(rate=10)


#%% Testing
if __name__=='__main__' and 0:
    
    lp.timer.start(50)
    
    for ii in range(100):
        lp.data=im[ii,:]
        #lp.update()
        time.sleep(.1)
        QtGui.QApplication.processEvents()
        
