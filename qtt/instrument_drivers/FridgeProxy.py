
#%%
from qcodes import Instrument
from zmqrpc.ZmqRpcClient import ZmqRpcClient
from functools impoprt partial

class FridgeProxy(Instrument):
    '''
    Generic proxy to fridge
    '''
    def __init__(self, name, ip_address = 'localhost', **kwargs):
        super().__init__(name, **kwargs)
        
        self.client = ZmqRpcClient(zmq_req_endpoints=["tcp://%s:30000" % ip_address])

        temperature = partial(self._proxy_wrapper, function_name='temperature',  default_value=-1)
        self.add_parameter('temperature', unit='K', get_cmd=temperature,
                           docstring='Temperature of cold plate')
        #_ = self.timestamp.get()
        
    def _proxy_wrapper(self,  function_name, default_value=None, time_out_waiting_for_response_in_sec = 3):
        try:
            r = self.client.invoke(function_name, time_out_waiting_for_response_in_sec=time_out_waiting_for_response_in_sec)
            return r
        except:
            return default_value
                

#%% Testing
if __name__=='__main__':
    fridge=FridgeProxy(name='fridgetest3', ip_address='localhost')
    
