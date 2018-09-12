from qupulse.serialization import StorageBackend


class StringBackend(StorageBackend):

    def __init__(self):
        self.__str = ''

    def put(self, identifier, data, overwrite=False):
        if overwrite:
            self.__str = data
            return
        self.__str += data

    def get(self, identifier):
        return self.__str

    def exists(self, identifier):
        return True
