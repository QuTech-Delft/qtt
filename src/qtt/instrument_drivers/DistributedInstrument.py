class DeprecationError(Exception):
    """ Raised for classes tha have been moved or removed."""


class InstrumentDataClient:
    def __init__(self, *_, **__):
        error_msg = f"{self.__class__.__name__} has been move to qtt.instrument_drivers.instrument_data_client"
        raise DeprecationError(error_msg)


class InstrumentDataServer:
    def __init__(self, *_, **__):
        error_msg = f"{self.__class__.__name__} has been move to qtt.instrument_drivers.instrument_data_server"
        raise DeprecationError(error_msg)
