
class Setting:

    def __init__(self, unit, value, minimum, maximum):
        self.unit = unit
        self.__value = value
        self.__minimum = minimum
        self.__maximum = maximum

    @property
    def value(self):
        return self.__value

    def in_range(self, value):
        return False if value < self.__minimum or value > self.__maximum else True

    def update_value(self, value):
        in_range = self.in_range(value)
        if in_range:
            self.__value = value
        return in_range
