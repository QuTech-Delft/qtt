from zmqrpc.ZmqRpcServer import ZmqRpcServerThread

from qtt.instrument_drivers.fridge_monitor import ConnectionSettings


class InstrumentDataServer:

    def __init__(self) -> None:
        """ Represents a server proxy for sending instrument measurable quantities to a client.

        The InstrumentDataServer contains the following instance attributes:
            settings: The connection settings of the server.
            rpc_functions: A dictionary with instrument functions and function names as keys.
                           The connected clients can call these functions after starting the server proxy.
        """
        self.settings = ConnectionSettings(address='*')
        self.rpc_functions = dict()

    def __create_rpc_server_thread(self) -> None:
        username = self.settings.username
        password = self.settings.password
        bind_addres = self.settings.tcp_bind_address()
        server = ZmqRpcServerThread(bind_addres,
                                    username=username,
                                    password=password,
                                    rpc_functions=self.rpc_functions)
        return server

    def run(self) -> None:
        """ Starts the server proxy and blocks the current thread.

        A keyboard interuption (CTRL+C) will stop and clean-up the server proxy.
        """
        server = self.__create_rpc_server_thread()
        print(' Enabled instrument server...')
        print(' Press CTRL+C to quit!')
        try:
            server.run()
            while True:
                continue
        except KeyboardInterrupt:
            print(' Done')
        finally:
            server.stop()
            server.join()
