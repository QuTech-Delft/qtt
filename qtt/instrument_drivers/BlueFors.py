
#%%
from qcodes import Instrument
from zmqrpc.ZmqRpcClient import ZmqRpcClient


class BlueFors(Instrument):
    '''
    Proxy to BlueFors fridge
    '''
    def __init__(self, name, ip_address = 'localhost', **kwargs):
        super().__init__(name, **kwargs)
        
        self.client = ZmqRpcClient(zmq_req_endpoints=["tcp://%s:30000" % ip_address])

        self.add_parameter('temperature', unit='mK', get_cmd=lambda : self.client.invoke('temperature'),
                           docstring='Temperature of cold plate')
        #_ = self.timestamp.get()

#%%
if __name__=='__main__':
    b=BlueFors(name='xld', ip_address='localhost')