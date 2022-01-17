from qupulse.pulses import FunctionPT, TablePT


class DataTypes:
    """ The possible data types for the pulse creation."""
    RAW_DATA = 'rawdata'
    QU_PULSE = 'qupulse'


class Templates:

    @staticmethod
    def chirp(name):
        """ Creates a chirp signal

        Args:
            name (str): The user defined name of the pulse template.

        Returns:
            FunctionPT: The pulse template with the chirp signal.
                     Parameters of the pulse template are the `duration` (in the same unit as time),
                     `omega_0` (in Hz), `delta_omega` (in Hz), `amplitude` and `phase`. Time is in ns.
                     """
        linear_chirp_template = FunctionPT(
            'amplitude*cos(2*pi*(omega_0+(t/(2*duration))*delta_omega) *t*1e-9+phase)', 'duration', channel=name)
        linear_chirp_template.__doc__ = 'Template for linear chirp\nAlso see https://en.wikipedia.org/wiki/Chirp\n\n'+linear_chirp_template.__doc__
        return linear_chirp_template

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
    def sawtooth(name, padding=0):
        """ Creates a sawtooth qupulse template for sequencing.

        Args:
            name (str): The user defined name of the sequence.
            padding (float): Padding to add at the end of the sawtooth

        Returns:
            TablePT: The sequence with the sawtooth wave.
        """
        tbl = [(0, 0), ('period*(1-width)/2', '-amplitude', 'linear'),
               ('period*(1-(1-width)/2)', 'amplitude', 'linear'),
               ('period', 0, 'linear')]
        if padding > 0:
            tbl += [(f'period+{padding}', 0, 'hold')]
        return TablePT({name: tbl})

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
        r""" Creates a skewed sawtooth qupulse template for sequencing.
        This pulse is symmetric, has total integral zero and right at period/2 it
        has amplitude 0 and a sharp corner.

        A visual representation of the waveform is:

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
        T is period and A is the amplitude. Negative amplitude will produce an inverted pulse.

        Args:
            name (str): The user defined name of the sequence.

        Returns:
            TablePT: The sequence with the skewed sawtooth wave.
                     Parameters of the pulse template are the `amplitude` and `period`.
        """
        return TablePT({name: [(0, 0),
                               ('period/6', 'amplitude', 'linear'),
                               ('period/3', '-amplitude', 'linear'),
                               ('period/2', 0, 'linear'),
                               ('period*2/3', '-amplitude', 'linear'),
                               ('period*5/6', 'amplitude', 'linear'),
                               ('period', 0, 'linear')]})

    @staticmethod
    def pulse_table(name, entries):
        return TablePT({name: entries})
