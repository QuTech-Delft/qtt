import logging
l = logging.getLogger()
l.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s (%(filename)s:%(lineno)d)')
try:
    l.handlers[0].setFormatter(formatter)
except:
    pass


#%% Kill all processes...

# make sure drivers are disconnected...
#os.system("taskkill /F /im python.exe")


#%% Use HDF5 format

try:
    from qcodes.data.hdf5_format import HDF5Format
    #qc.DataSet.default_formatter=HDF5Format()
except:
    pass

