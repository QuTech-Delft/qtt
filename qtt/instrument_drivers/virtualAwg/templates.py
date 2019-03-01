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
    def rollover_marker(name):
        """Creates a TTL pulse qupulse template for sequencing that rolls over to the subsequent period.

            ---------         ----------
                     |        |
                     |        |
                     ----------
            <---------period------------>
            <-----offset----->
            <--------> uptime <--------->

        Args:
            name (str): The user defined name of the sequence.

        Returns:
            TablePT: The sequence with the marker pulse and rollover part of the pulse.
        """
        return TablePT({name: [(0, 1),
                               ('offset + uptime - period', 0),
                               ('offset', 1),
                               ('period', 1)]})

    @staticmethod
    def skewed_sawtooth(name):
        """ Creates a skewed sawtooth qupulse template for sequencing.
        This pulse is symmetric, has total integral zero and right at T/2 it
        has amplitude 0 and a sharp corner.

          A     /\              /\
               /  \            /  \
          0   /    \    /\    /    \
                    \  /  \  /
         -A          \/    \/
               T/6
              <->
                 T/3
              <------>
                  T/2
              <--------->
                         T
              <-------------------->

        Args:
            name (str): The user defined name of the sequence.

        Returns:
            TablePT: The sequence with the skewed sawtooth wave.
        """
        return TablePT({name: [(0, 0),
                                 ('T/6', 'A', 'linear'),
                                 ('T/3', '-A', 'linear'),
                                 ('T/2', 0, 'linear'),
                                 ('T*2/3', '-A', 'linear'),
                                 ('T*5/6', 'A', 'linear'),
                                 ('T', 0, 'linear')]})
