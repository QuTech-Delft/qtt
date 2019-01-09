from qupulse.pulses import TablePT


class DataTypes:
    """ The possible data types for the pulse creation."""
    RAW_DATA = 'rawdata'
    QU_PULSE = 'qupulse'


class Templates:

    @staticmethod
    def square(name):
        """ Creates a block wave qupulse template for sequencing.

        Args:
            name (str): The user defined name of the sequence.

        Returns:
            TablePT: The template with the square wave.
        """
        return TablePT({name: [(0, 0), ('period/4', 'amplitude'),
                               ('period*3/4', 0), ('period', 0)]})

    @staticmethod
    def sawtooth(name):
        """ Creates a sawtooth qupulse template for sequencing.

        Args:
            name (str): The user defined name of the sequence.

        Returns:
            TablePT: The sequence with the sawtooth wave.
        """
        return TablePT({name: [(0, 0), ('period*(1-width)/2', '-amplitude', 'linear'),
                               ('period*(1-(1-width)/2)', 'amplitude', 'linear'),
                               ('period', 0, 'linear')]})

    @staticmethod
    def hold(name):
        """Creates a DC offset qupulse template for sequencing.

        Args:
            name (str): The user defined name of the sequence.

        Returns:
            TablePT: The sequence with the wait pulse.
        """
        return TablePT({name: [(0, 'offset'), ('period', 'offset')]})

    @staticmethod
    def marker(name):
        """Creates a TTL pulse qupulse template for sequencing.

        Args:
            name (str): The user defined name of the sequence.

        Returns:
            TablePT: The sequence with the wait pulse.
        """
        return TablePT({name: [(0, 0), ('offset', 1),
                               ('offset+uptime', 0), ('period', 0)]})

    @staticmethod
    def pulse_table(name, entries):
        return TablePT({name: entries})
