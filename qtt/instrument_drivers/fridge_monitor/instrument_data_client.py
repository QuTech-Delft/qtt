from functools import partial
from typing import Any

from qcodes import Instrument
from zmqrpc.ZmqRpcClient import ZmqRpcClient

from qtt.instrument_drivers.fridge_monitor import ConnectionSettings


class InstrumentDataClient(Instrument):

    def __init__(self, name: str, **kwargs) -> None:
        """ Represents a proxy client for collecting instrument measurable quantities from a InstrumentDataServer.

        The InstrumentDataClient contains the following instance attributes:
            settings: The connection settings of the server.
            timeout: The timeout waiting time in seconds for responce of a remote procedure call.

        Args:
            name (str): the name of the qcodes instrument.
        """
        super().__init__(name, **kwargs)
        self.settings = ConnectionSettings(address='localhost')
        self.__client = None
        self.timeout = 5

    def __create_rpc_server_client(self) -> None:
        username = self.settings.username
        password = self.settings.password
        bind_addres = [self.settings.tcp_bind_address()]
        self.__client = ZmqRpcClient(bind_addres, username=username, password=password)

    def connect(self) -> None:
        """Connects the client to the server using the settings."""
        self.__create_rpc_server_client()

    def __invoke_getter(self, function_name, default_return_value, timeout=None):
        if self.__client is None:
            raise AttributeError('Client not connected! Run connect first.')
        try:
            if timeout==None:
                timeout = self.timeout
            return self.__client.invoke(function_name, None, timeout)
        except Exception:
            return default_return_value

    def __invoke_setter(self, function_name, value, parameter_name='argument', timeout=None):
        function_parameters = {parameter_name: value}
        if self.__client is None:
            raise AttributeError('Client not connected! Run connect first.')
        if timeout==None:
            timeout = self.timeout
        return self.__client.invoke(function_name, function_parameters, timeout)

    def add_get_set_parameter(self, name: str, default_return_value: Any = None, timeout: Any = None, **kwargs) -> None:
        get_command = partial(self.__invoke_getter, name, default_return_value=default_return_value, timeout=timeout)
        set_command = partial(self.__invoke_setter, name, timeout=timeout)
        self.add_parameter(name, get_cmd=get_command, set_cmd=set_command, **kwargs)

    def add_get_parameter(self, function_name: str, default_return_value: Any = None, timeout: Any = None, **kwargs) -> None:
        """ Creates a new get parameter for the instrument client.

        Args:
            function_name: The name of the instrument get parameter.
            unit: The unit of the instrument get parameter.
            default_value: The initial value and on error return value for the get parameter.
            docstring: The get parameter documentation.
        """
        get_command = partial(self.__invoke_getter, function_name, default_return_value, timeout=timeout)
        self.add_parameter(function_name, get_cmd=get_command, **kwargs)
