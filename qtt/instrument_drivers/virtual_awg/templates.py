from qctoolkit.pulses import TablePT


class DataTypes:
    """ The possible data types for the pulse creation."""
    RAW_DATA = 'RAW_DATA'
    QC_TOOLKIT = 'QC_TOOLKIT'


class Templates:

    @staticmethod
    def square(name):
        """ Creates a block wave QC toolkit template for sequencing.

        Arguments:
            name (str): The user defined name of the sequence.

        Returns:
            The template with the square wave.
        """
        return TablePT({name: [(0, 0), ('period/4', 'amplitude'), ('period*3/4', 0), ('period', 0)]})

    @staticmethod
    def sawtooth(name):
        """ Creates a sawtooth QC toolkit template for sequencing.

        Arguments:
            name (str): The user defined name of the sequence.

        Returns:
            The sequence with the sawtooth wave.
        """
        return TablePT({name: [(0, 0), ('period/4', 'amplitude', 'linear'),
                               ('period*3/4', '-amplitude', 'linear'), ('period', 0, 'linear')]})

    @staticmethod
    def hold(name):
        """Creates a DC offset QC toolkit template for sequencing.

        Arguments:
            name (str): The user defined name of the sequence.

        Returns:
            The sequence with the wait pulse.
        """
        return TablePT({name: [(0, 'offset'), ('holdtime', 'offset')]})

    @staticmethod
    def marker(name):
        """Creates a TTL pulse QC toolkit template for sequencing.

        Arguments:
            name (str): The user defined name of the sequence.

        Returns:
            The sequence with the wait pulse.
        """
        return TablePT({name: [(0, 1), ('period*uptime', 0), ('period', 0)]})