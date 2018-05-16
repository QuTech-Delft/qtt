""" Mathematical functions and models """

import numpy as np
import qtt.pgeometry
import qcodes
import copy
import matplotlib.pyplot as plt

from qcodes import MatPlot

def get_dataset(ds):
    """ Get a dataset from a results dict, a string or a dataset """    
    if isinstance(ds, dict):
        ds=ds.get( 'dataset', None)
    if ds is None:
        return None
    if isinstance(ds, str):
        ds = qcodes.load_data(ds)        
    return ds


def analyse_awg_to_plunger(result, method='hough', fig=None):

    ds=get_dataset(result)

    if method=='hough':
        import cv2
        im, tr = qtt.data.dataset2image(ds)
        imextent = tr.scan_image_extent()
        ims, r=qtt.algorithms.images.straightenImage(im, imextent, mvx=3, mvy=2)
        H=r[4]
        
        gray=qtt.pgeometry.scaleImage(ims)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        
        lines = cv2.HoughLines(edges,1,np.pi/180,int(gray.shape[0]*.8 ))
        
        a=lines[:,0,1]
        angle_pixel=np.percentile(a, 50)+0*np.pi/2
        
        fac=2
        
        xpix=np.array([[0,0],[-fac*np.sin(angle_pixel),fac*np.cos(angle_pixel)]]).T
        tmp=qtt.pgeometry.projectiveTransformation(np.linalg.inv(H), xpix)
        xscan=tr.pixel2scan( tmp)

        
        #p0=tr.pixel2scan( np.array([[0],[0]]))
        
        def vec2angle(v):
            return np.arctan2(v[0], v[1])
        angle=vec2angle(xscan[:,1]-xscan[:,0])
        
        
    elif method=='click':
        raise Exception('not implemented')
    else:
        raise Exception('method %s not implemented' % (method,) )
        
            
    result=copy.deepcopy(result)
    result['_angle_pixel']=angle_pixel
    result['angle']=angle
    result['awg_to_plunger_correction']=np.arctan(angle)

    if fig is not None:
        plt.figure(fig-1); plt.clf()
        MatPlot(ds.default_parameter_array(), num=fig-1)
        #qtt.pgeometry.plotPoints(xscan, '.m')
        
        if 0:
            yy=[]
            for ii in np.arange(-1,2):
                theta=angle_pixel
                c = np.cos(theta)
                s = np.sin(theta)
                x0 = -s*ii
                y0 = c*ii
                xpix=np.array([[x0],[y0]])
                tmp=qtt.pgeometry.projectiveTransformation(np.linalg.inv(H), xpix)
                xscan=tr.pixel2scan( tmp)
                #qtt.pgeometry.plotPoints(xscan, '.r')
                yy+=[xscan]
            
        v=xscan
        rho=v[0]*np.cos(angle)-np.sin(angle)*v[1]
        qtt.pgeometry.plot2Dline([np.cos(angle), -np.sin(angle),-rho], 'm', label='angle')

        plt.figure(fig); plt.clf()
        plt.imshow(gray); plt.axis('image')
        for ii in np.arange(-20,20):
            theta=angle_pixel
            c = np.cos(theta)
            s = np.sin(theta)
            x0 = -s*ii
            y0 = c*ii
            plt.plot(x0, y0, '.r')
    
        for offset in [-20,0,20]:
            label=None
            if offset is 0:
                label='detected angle' 
            qtt.pgeometry.plot2Dline([np.cos(angle_pixel), np.sin(angle_pixel), offset], 'm', label=label)
        plt.title('Detected line direction')

        plt.figure(fig+1); plt.clf()
        plt.imshow(edges); plt.axis('image')
        plt.title('Detected edge points')

    return result
        
def plot_awg_to_plunger(result, fig=10):
    angle=result['angle']
    
    ds=get_dataset(result)
    plt.figure(fig); plt.clf()
    MatPlot(ds.default_parameter_array(), num=fig)
    for offset in [-20,0,20]:
        label=None
        if offset is 0:
            label='detected angle' 
        qtt.pgeometry.plot2Dline([np.cos(angle), np.sin(angle), offset], 'm', label=label)
    plt.title('Detected line direction')
    

# TODO
#
# 1. Clean up code
# 2. Add documentation
# 3. Add unit test
# 4. Add example
# 5. Add method 'click' to manually determine the awg_to_plunger
# 6. Use code on V1
    
    
#