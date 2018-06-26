""" Functionality to determine the awg_to_plunger ratio

pieter.eendebak@tno.nl

"""

#%% Load packages
import numpy as np
import qtt.pgeometry
import qcodes
import copy
import matplotlib.pyplot as plt

from qcodes import MatPlot
import qtt.algorithms.images
import qtt.utilities.imagetools

#%% Helper functions
def get_dataset(ds):  
    """ Get a dataset from a results dict, a string or a dataset
    
    Args:
        ds (dict, str or qcodes.DataSet): the data to be put in the dataset

    Returns:
        ds (qcodes.DataSet) dataset made from the input data
        
    """
    if isinstance(ds, dict):
        ds = ds.get('dataset', None)
    if ds is None:
        return None
    if isinstance(ds, str):
        ds = qcodes.load_data(ds)
    return ds

#%% Main functions

def measure_awg_to_plunger(station, gate, minstrument, sweeprange=100, method='hough'):
    """ Perform measurement for awg to plunger using 2D fast scan check?
    
    Args:
        station (str): name of station
        gate (int): plunger gate number
        minstrument: 
        sweeprange (int): sweeprange in mV?
        method: Not used
            
    Returns:
        result: (qcodes.DataSet) 
    
    """
    from qtt.measurements.scans import scanjob_t
    gates=station.gates
    value0=gates.get(gate)
    scanjob=scanjob_t({'sweepdata': {'param': gate, 'range': sweeprange}, 'stepdata':{'param': gate, 'range':100, 'step': 3, 'start': value0-sweeprange/2} })
    scanjob['minstrument']=minstrument
    scanjob['minstrumenthandle']=minstrument
    scanjob['Naverage']=100
    scanjob.setWaitTimes(station)
    scanjob['wait_time_startscan']+=.2
    
    ds = qtt.measurements.scans.scan2Dfast(station, scanjob)
    result = {'type': 'awg_to_plunger', 'awg_to_plunger': None, 'dataset': ds.location}
    return result


def analyse_awg_to_plunger(result, method='hough', fig=None):
    """ Determine the awg_to_plunger ratio from a scan result
    
    Args:
        result: (qcodes.DataSet) result of measure_awg_to_plunger
        method: (str) image processing transform method, only hough supported
        fig: (str, int or None) figure number or name, if None no plot is made
        
    Returns:
        result:
    
    """
    assert(result.get('type')=='awg_to_plunger')
    ds = get_dataset(result)

    if method == 'hough':
        import cv2
        im, tr = qtt.data.dataset2image(ds)
        imextent = tr.scan_image_extent()
        istep=tr.istep()
        ims, r = qtt.algorithms.images.straightenImage(
            im, imextent, mvx=istep, mvy=None)
        H = r[4]

        gray = qtt.pgeometry.scaleImage(ims)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi/180, int(gray.shape[0]*.8))

        if lines is None:
            angle_pixel=None
            angle=None
            xscan=None
        else:
            a = lines[:, 0, 1]
            angle_pixel = np.percentile(a, 50)+0*np.pi/2
    
            fac = 2
    
            xpix = np.array(
                [[0, 0], [-fac*np.sin(angle_pixel), fac*np.cos(angle_pixel)]]).T
            tmp = qtt.pgeometry.projectiveTransformation(np.linalg.inv(H), xpix)
            xscan = tr.pixel2scan(tmp)
    
            #p0=tr.pixel2scan( np.array([[0],[0]]))
    
            def vec2angle(v):
                return np.arctan2(v[0], v[1])
            angle = vec2angle(xscan[:, 1]-xscan[:, 0])

    elif method == 'click':
        raise Exception('not implemented')
    else:
        raise Exception('method %s not implemented' % (method,))

    result = copy.copy(result)
    result['_angle_pixel'] = angle_pixel
    result['angle'] = angle
    if angle is None:
        result['awg_to_plunger_correction'] = None
    else:
        scanratio = tr.istep_step()/tr.istep_sweep()
        print('scanratio: %.3f' % scanratio)
        result['awg_to_plunger_correction'] = scanratio*np.tan(angle)

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        MatPlot(ds.default_parameter_array(), num=fig)

        if 0:
            yy = []
            for ii in np.arange(-1, 2):
                theta = angle_pixel
                c = np.cos(theta)
                s = np.sin(theta)
                xpix = np.array([[-s*ii], [c*ii]])
                tmp = qtt.pgeometry.projectiveTransformation(
                    np.linalg.inv(H), xpix)
                xscan = tr.pixel2scan(tmp)
                yy += [xscan]

        if xscan is not None:
            v = xscan
            rho = v[0]*np.cos(angle)-np.sin(angle)*v[1]
            qtt.pgeometry.plot2Dline(
                [np.cos(angle), -np.sin(angle), -rho], 'm', label='angle')

        plt.figure(fig+1)
        plt.clf()
        plt.imshow(gray)
        plt.axis('image')

        if angle_pixel is not None:
            for offset in [-20, 0, 20]:
                label = None
                if offset is 0:
                    label = 'detected angle'
                qtt.pgeometry.plot2Dline(
                    [np.cos(angle_pixel), np.sin(angle_pixel), offset], 'm', label=label)
        if angle is not None:
            plt.title('Detected line direction: angle %.2f' % (angle,))

        plt.figure(fig+2)
        plt.clf()
        plt.imshow(edges)
        plt.axis('image')
        plt.title('Detected edge points')

    return result


def plot_awg_to_plunger(result, fig=10):
    """ This function tests the analyse_awg_to_plunger function. Plotting is optional and for debugging purposes.
    
    Args:
        result:
        fig (int): index of matplotlib window

        
    """

    if not result.get('type', None)=='awg_to_plunger':
        raise Exception('calibration result not of correct type ')
        
    angle = result['angle']

    ds = get_dataset(result)
    plt.figure(fig)
    plt.clf()
    MatPlot(ds.default_parameter_array(), num=fig)
    if angle is not None:
        for offset in [-20, 0, 20]:
            label = None
            if offset is 0:
                label = 'detected angle'
            qtt.pgeometry.plot2Dline(
                [np.cos(angle), -np.sin(angle), offset], 'm', label=label)
    plt.title('Detected line direction')


# %% Test functions

def test_awg_to_plunger(fig=None):
    """ Plot results of awg_to_plunger calibration check?
    
    Args:
        fig (str, int or None): default None. Name of figure to plot in, if None not plotted
        
    Returns:
        Nothing
        
    """
    x = np.arange(0, 80, 1.0).astype(np.float32)
    y = np.arange(0, 60).astype(np.float32)
    z = np.meshgrid(x, y)
    z = 0.01 * z[0].astype(np.uint8)
    angle = -0.7 * (np.pi/4.0)
    qtt.utilities.imagetools.semiLine(z, np.array([[0], [y.max()]]), angle, w=2.2, l=30, H=0.52)
    ds = qtt.data.makeDataSet2Dplain('x', x, 'y', y, 'z', z)
    result = {'dataset': ds, 'type': 'awg_to_plunger'}
    r = analyse_awg_to_plunger(result, method='hough', fig=fig)
    if fig:
        print(r)
        print('ange input %.3f: fit %s' % (angle, str(r['angle'])))


if __name__ == '__main__':
    test_awg_to_plunger(fig=10)



        