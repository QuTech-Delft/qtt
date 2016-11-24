import numpy as np

def FermiLinear(x, a, b, cc, A, T, kb=1):
    """ Fermi distribution with linear function added 
    
    Arguments:
        x (numpy array): independent variable
        a, b (float): coefficients of linear part
        cc (float): center of Fermi distribution
        A (float): amplitude of Fermi distribution
        T (float): temperature Fermi distribution
        kb (float, default: 1): temperature scaling factor

    Returns:
        y (numpy array): value of the function
        
    .. math::
        
        y = a*x + b + A*(1/ (1+\exp( (x-cc)/(kb*T) ) ) )
        

    """   
    y = a*x+b+A*1./(1+np.exp( (x-cc)/(kb*T) ) )
    return y

