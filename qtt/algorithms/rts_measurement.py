# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:45:18 2018

@author: riggelenfv

These are functions which can be used to measure random telegraph signal (RTS). For these functions to work,
a station must defined, the sample needs to be tuned to a clear anticrossing, 
"""

#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
import qcodes
import qtt
from qcodes import MatPlot
from qtt.measurements.scans import scan2Dturbo, scanjob_t, scan1Dfast
from qtt.data import dataset2image
from qtt.algorithms.fitting import FermiLinear, fitFermiLinear
from qtt.tools import addPPTslide, update_dictionary
from anticross_fitting import fit_anticrossing
from projects.calibration_manager.calibration_master import StorageMaster
from qtt.data import makeDataSet1Dplain

#%% making a 2Dscan of the anti-crossing
def scan2Dturbo_anticrossing(station, plungers, minstrument, minstrumenthandle, scan_range = 40, Naverage = 100): 
    """Defines a scanjob for a scan2Dturbo, which profides the dataset for the fitting of the anticrossing"""
       
    if hasattr(station, 'RF'):
            RF.on()
    scanjob = scanjob_t({'resolution': [ 96,96]})    
    scanjob = scanjob_t()
    scanjob['wait_time_startscan'] = 5
    scanjob['minstrument'] = minstrument
    scanjob['minstrumenthandle']= minstrumenthandle 
    scanjob['sweepdata'] = {'param': plungers[0], 'range': scan_range}
    scanjob['stepdata'] = {'param': plungers[1], 'range': scan_range}
    scanjob['Naverage'] = Naverage
    dataset, _, _ = scan2Dturbo(station, scanjob)
    
    if hasattr(station, 'RF'):
        RF.off()

    return dataset

#%%

def pixeltovaluev(image,dataset,p):
    """Takes point (p) in a 2d image of which the coordinates are in pixels and calculates the coordinates in mV, 
    assuming that the dimensions of the 2d image are the same as for the dataset which is give. """
    
    patch_shape = image['patch'].shape
    im, tr = dataset2image(dataset, arrayname='measured')
    tr.scan_image_extent()
    valuev = np.hstack(((tr.extent[1]-tr.extent[0])*p[0]/patch_shape[0]+tr.extent[0], \
                  (tr.extent[3]-tr.extent[2])*p[1]/patch_shape[1]+tr.extent[2]))
    
    return valuev


def position_in_anticrossing(point1,point2, anticross_fit, dataset, dv):
    """Determines a point in the charge stability diagram on the (0,0)-(0,1) (or similar) addition line,
    a distance dv (mV) from the anticrossing, measured along that line. """
    
    #coordinates of these points in the charge stability diagram
    p1v = pixeltovaluev(anticross_fit,dataset,point1)
    p2v = pixeltovaluev(anticross_fit,dataset,point2)
    
    #dv 'distance' in mV from the anti crossing along the (0,0)-(0,1) addition line
    dydx = (p1v[1]-p2v[1])/(p1v[0]-p2v[0])
    dx = dv/(np.sqrt(dydx**2+2))
    dy = dx*dydx
    
    p3v = [p1v[0]+dx, p1v[1]+dy] #coordinates of the point on the (0,0)-(0,1) addition line, dv from the anti crossing
    
    return p3v

#%%

def scan1Dfast_additionline(station, plungers, minstrument,minstrumenthandle, scan_range, Naverage = 256):
    """Defines a scanjob for a scan1Dfast, which profides the dataset for the fitting of the addition line. """
    
    if hasattr(station, 'RF'):
            RF.on()
            
    station.awg.delay_FPGA=20e-7    
    scanjob = scanjob_t()
    scanjob['wait_time_startscan'] = 5
    scanjob['minstrument'] = minstrument
    scanjob['minstrumenthandle']= minstrumenthandle 
    scanjob['sweepdata'] = {'param': plungers[0], 'range': scan_range}
    scanjob['Naverage'] = Naverage
    dataset = scan1Dfast(station, scanjob)
    
    if hasattr(station, 'RF'):
        RF.off()
    
    return dataset

#%%

def fit_addition_line(dataset):
    """Fits a FermiLinear function to the addition line and the middle of the step."""
    
    y_array = dataset.default_parameter_array()
    setarray = y_array.set_arrays[0]
    xdata = np.array(setarray)
    ydata = np.array(y_array)
    
    # fitting of the FermiLinear function
    pp = fitFermiLinear(xdata, ydata, verbose=1, fig=None)
    pfit = pp[0]
    p0 = pp[1]['p0']
    
    y0 = FermiLinear(xdata, *list(p0) )
    y = FermiLinear(xdata, *list(pfit) )
    
    return pfit, y0, y, setarray

#%%

def measure_RTS_fpga(station, plungers, period = 8e-3):
    """Peforms the RTS measurement when the measurement instrument is a FPGA. This is done by simply recording the
    value of the sensing dot while staying on (or very close to) the addition line. """
    
    if hasattr(station, 'RF'):
            RF.on()
            
    samplerate = fpga.sampling_frequency.get()
    station.awg.sweep_gate(plungers[0], 0, period)
    qtt.time.sleep(4)
    
    FPGA_mode = 0
    
    fpga.set('mode', FPGA_mode)
    _, data_fpga, _ = fpga.readFPGA(ReadDevice=['FPGA_ch1'])
    
    station.awg.stop()
    
    if hasattr(station, 'RF'):
        RF.off()
    # store the data
    data = data_fpga[1:]
    
    return data, samplerate


def measure_RTS_digitizer(station, samplerate = 2e6, mV_range=2000, memsize=16*115000, posttrigger_size=16):
    """Peforms the RTS measurement when the measurement instrument is a digitizer. This is done by simply recording the
    value of the sensing dot while staying on (or very close to) the addition line. """
    
    station.digitizer.sample_rate(samplerate)
    samplerate = station.digitizer.sample_rate()
    qtt.time.sleep(4)

    data=station.digitizer.single_software_trigger_acquisition(mV_range=mV_range, memsize=memsize, posttrigger_size=posttrigger_size)
    
    return data, samplerate

#%% function for measuring RTS, starting from a centered anti-crossing


def measurement_RTS(plungers,filename, station, minstrument, minstrumenthandle, dv =10, mrange=1.5, step=0.1, nmeasurements=5, memsize=10*128*1024, plungervalues = None, fig = None):
    """
    This function takes as starting point a clear anticrossing, fit the anti-crossing, determines from there a position in the charge 
    stability diagram, fits the addition line in two steps and performes a RTS measurement. For this to work, the anti-crossing must look 
    neat and the tunneling frequency should be sufficiently low.
    
    Args:
        plungers ([str, str]): names of the two plungers of the double dot that form the anticrossing, plungers[0] is the plungers of the dot
            for which the random telegraph signal is measured between the dot and the reservoir.
        filename (str): name the datafile of the RTS measurement should get.
        station (QCoDeS station): measurement setup
        minstrument ([int]): array with one int as element, this sets the channel of the measurement instrument for the fast and turbo scan
        minstrumenthandle (str): measurement instrument for the fast and turbo scan
        dv (int): distance in mV from the anticrossing, along the (0,1) addition line, where the addition line is fitted
        mrange (float): total range in mV for the plunger gate around the addition line for which RTS measurements are taken
        step (float): size of the step in mV with which the plunger gate is changed
        nmeasurements (int): number of RTS measurements performed for a certain set of gate values
        memsize (int): size of the file in which the data is saved, should be of the form n*128*1024
        plungervalues (None or [float, float]): the place in the charge stability diagram where the addition line is fitted,
            profides a short cut in the function, if not None the step of fitting the anticrossing and deterining the
            position is skipped.
        fig (None or int): shows figure and sends them to the ppt if not None
        
    Returns:
        datafile (str): full name of the datafile where the series of RTS measurements is saved
        dictionary with relevent (fit) parameters
    
    
    """
    #some constants
    kb = 86 # uev/K
    kT = 75e-3*kb # ueV, estimate
    la = 80 # ueV/mV, estimate
    
    #short cut, setting the plunger values by hand
    if plungervalues:
        gates.set(plungers[0], plungervalues[0])
        gates.set(plungers[1], plungervalues[1])
        
    else:
        #making a 2d scan of the anti-crossing    
        dataset = scan2Dturbo_anticrossing(station, plungers, minstrument, minstrumenthandle, scan_range = 40, Naverage = 100)
        
        #fitting the anti-crossing
        anticross_fit = fit_anticrossing(dataset, psi=np.pi/4, verbose=0)
            
        if fig:
            anticross_fit = fit_anticrossing(dataset, psi=np.pi/4, plot=True, verbose=0)
            plt.xlabel(plungers[0])
            plt.ylabel(plungers[1])
            plt.suptitle(dataset.location)
            addPPTslide(title='Fit anti-crossing', fig=anticross_fit['fig_num'], notes=str(gates.allvalues()))
        
        # two points in the fit that describe the (0,0)-(0,1) transition line
        point1 = np.array(anticross_fit['cdata'][1])
        point2 = anticross_fit['cdata'][4][0]
        
        #choosing a point on this addition line, dv (mV) from the anticrossing
        p3v = position_in_anticrossing(point1, point2, anticross_fit, dataset, dv = dv)
        
        gates.set(plungers[0], p3v[0])
        gates.set(plungers[1], p3v[1])
            
    
    #fine tuning the plunger value by measuring and fitting the addition line, first for a larger range 
    dataset = scan1Dfast_additionline(station, plungers, minstrument, minstrumenthandle, scan_range=20, Naverage = 256)
    pfit, y0, y, setarray = fit_addition_line(dataset)
    
    if fig:
        plot = MatPlot(dataset.default_parameter_array() )
        v0 = qcodes.DataArray(name='fit', label='fit', preset_data=y0,  set_arrays=(setarray,) )
        plot.add(v0, alpha=.2, label='initial guess')                  
        v = qcodes.DataArray(name='fit', label='fit', preset_data=y,  set_arrays=(setarray,) )
        plot.add(v, label='fitted curve')
        plt.legend()
        addPPTslide(title='Fitted charging line', fig=plot.fig.number, notes=str(gates.allvalues()))
    
    # change the gate value to centre of the charge addition line
    gates.set(plungers[0], pfit[2])
    
    # repeat for more accurate determination of centre of addition line
    fine_rang = kT/la*20
    
    dataset_fine = scan1Dfast_additionline(station, plungers, minstrument, minstrumenthandle, scan_range=fine_rang, Naverage = 256)
    pfit_fine, y0_fine, y_fine, setarray_fine = fit_addition_line(dataset_fine)
    
    if fig:
        plot_fine = MatPlot(dataset_fine.default_parameter_array() )
        v0 = qcodes.DataArray(name='fit', label='fit', preset_data=y0_fine,  set_arrays=(setarray_fine,) )
        plot_fine.add(v0, alpha=.2, label='initial guess')                  
        v = qcodes.DataArray(name='fit', label='fit', preset_data=y_fine,  set_arrays=(setarray_fine,) )
        plot_fine.add(v, label='fitted curve')
        plt.legend()
        addPPTslide(title='Fitted charging line (fine)', fig=plot_fine.fig.number, notes=str(gates.allvalues()))

    gates.set(plungers[0], pfit_fine[2])
        
    #to save the data at the right location and name
    datafile=os.path.join(qcodes.DataSet.default_io.base_location, filename)
    storage= StorageMaster(name='rts' , storagefile=datafile)
    
    
    #measuring RTS 
    p=gates.get(plungers[0])
    plungervalues = []
    
    for ii, dp in enumerate(np.arange(-mrange/2,mrange/2,step)):
        gates.set(plungers[0],p+dp)
        print(gates.get(plungers[0]))
        plungervalues.append(gates.get(plungers[0]))
        
        qtt.time.sleep(4)

        for ii in range(nmeasurements):
            print(ii)
                
            if hasattr(station, 'digitizer'):
                data, samplerate = measure_RTS_digitizer(station, samplerate = 2e6, mV_range=2000, memsize=10*115000, posttrigger_size=16)
            
            if hasattr(station, 'fpga'):
                data, samplerate = measure_RTS_fpga(station, plungers, period = 8e-3)
            
            #saving the data as qucode datasets within a storage object
            set_vals = np.arange(0, len(data), 1)
            dataset_rts = makeDataSet1Dplain('index', set_vals, yname='measured', y=data,location = False, loc_record={'label': 'RTS'})
            update_dictionary(dataset_rts.metadata, station=station.snapshot(), scantime=str(datetime.datetime.now()), allgatevalues=gates.allvalues(), samplerate=samplerate)
            dataset_rts.write(write_metadata=True)
            
            ts=qtt.data.dateString().strip()
            storage.save_result(dataset_rts, ts)
            
    
    return datafile, {'plungers':plungers, 'sampling rate':samplerate, 'plunger values':plungervalues,\
                      '1st guess position on addition line':p3v, 'fit parameters addition line':[pfit, y0, y, setarray],\
                      'fit parameters addition line fine': [pfit_fine, y0_fine, y_fine, setarray_fine]}