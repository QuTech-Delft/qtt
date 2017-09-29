
#%%
import time
from qcodes import Instrument
from zmqrpc.ZmqRpcClient import ZmqRpcClient


class BlueFors(Instrument):
    '''
    Proxy to BlueFors fridge
    '''
    def __init__(self, name, ip_address = 'localhost', **kwargs):
        super().__init__(name, **kwargs)
        # we need this to be a parameter (a function does not work with measure)
        
        self.client = ZmqRpcClient(zmq_req_endpoints=["tcp://%s:30000" % ip_address])


        t=self.client.invoke('print_time')


        self.add_parameter('temperature', unit='mK', get_cmd=lambda : self.client.invoke('temperature'),
                           docstring='Temperature of cold plate')
        #_ = self.timestamp.get()

#%%

b=BlueFors(name='xld', ip_address='localhost')