from functools import partial

from qcodes import Instrument
from zmqrpc.ZmqRpcClient import ZmqRpcClient


class InstrumentDataClient(Instrument):
    '''
    A proxy client for collecting instrument measurable quantities
    from a server.

    Args:
        name (str): the name of the instrument.
        address (str): the ip-address of the server.
        port (int): the port number of the proxy server.
        user (str): a username for protection.
        password (str): a password for protection.
    '''

    def __init__(self, name, address='localhost', port=8080, user=None,
                 password=None, **kwargs):
        super().__init__(name, **kwargs)
        self._client_ = ZmqRpcClient(["tcp://{0}:{1}".format(address, port)],
                                     username=user, password=password)

    def __repr__(self):
        return '<{} at %x{}: name {}>'.format(self.__class__, '%x' % id(self), self.name)

    def __proxy_wrapper__(self, command_name, default_value, params,
                          sec_time_out=3):
        try:
            return self._client_.invoke(command_name, params, sec_time_out)
        except:
            return default_value

    def add_measurable_quantity(self, name='quantity', unit='arb.',
                                default_value=None, doc_string='Unknown',
                                command_name='', params=None):
        '''Adds a instument function to the dataclient.'''
        if not command_name:
            command_name = name
        command = partial(self.__proxy_wrapper__, command_name=command_name,
                          default_value=default_value, params=params)
        self.add_parameter(name, unit=unit, get_cmd=command,
                           docstring=doc_string)
