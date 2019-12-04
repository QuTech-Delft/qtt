from zmqrpc.ZmqRpcServer import ZmqRpcServerThread


class InstrumentDataServer:
    """ Represents a server proxy for sending instrument measurable quantities
    to a client.

    Args:
        functions (dict): the instrument functions.
        address (str): the ip-address of the server.
        port (int): the port number of the proxy server.
        user (str): a username for protection.
        password (str): a password for protection.
    """

    def __init__(self, functions, address='*', port=8080, user=None,
                 password=None):
        self._server_ = ZmqRpcServerThread("tcp://{0}:{1}".format(
            address, port),
            rpc_functions=functions,
            username=user,
            password=password)

    def __repr__(self):
        return '<{} at %x{}: is_alive {}>'.format(type(self), '%x' % id(self), self._server_.is_alive())

    def run(self):
        """Starts the server proxy and blocks the current thread. A keyboard
        interuption will stop and clean-up the server proxy."""
        print(' Enabled instrument server...')
        print(' Press CTRL+C to quit!')
        try:
            self._server_.start()
            while (True):
                continue
        except KeyboardInterrupt:
            print(' Done')
        finally:
            self._server_.stop()
            self._server_.join()

    def start(self):
        """Starts the server proxy."""
        self._server_.start()

    def stop(self):
        """Stops the server proxy."""
        self._server_.stop()
