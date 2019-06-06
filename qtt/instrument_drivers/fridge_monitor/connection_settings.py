from typing import Optional


class ConnectionSettings:

    def __init__(self, address: str, port: int = 8888, username: Optional[str] = None, password: Optional[str] = None):
        """ Contains the required connection settings for the instrument data server and client."""
        self.address = address
        self.port = port
        self.username = username
        self.password = password

    def tcp_bind_address(self):
        """ Returns the TCP connection address."""
        return f'tcp://{self.address}:{self.port}'
