import numpy as np
import warnings
import logging
import copy
import numbers

from scipy import ndimage

import qcodes
import qtt
from qtt.measurements.scans import makeDataset_sweep, makeDataset_sweep_2D
from qtt.measurements.acquisition.interfaces import AcquisitionScopeInterface

from abc import ABC, abstractmethod


class VideoModeProcessor(ABC):
    """ Base class for VideoMode processing functionality """

    @abstractmethod
    def initialize(self, videomode):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def measure(self, videomode):
        return None

    @abstractmethod
    def scan_dimension(self):
        """ Return the dimension of the data to be shown (1 or 2) """

    @abstractmethod
    def process(self, measurement_data, videomode):
        return measurement_data

    @classmethod
    def extend_videomode_name(self, name):
        """ String to append to VideoMode name """
        return name

    @classmethod
    def ppt_notes(self):
        """ Generate notes to be added in a powerpoint slide """
        return ''

    @classmethod
    def default_processing(self, measurement_data, videomode):
        diff_sigma = videomode.diffsigma

        dd = []
        for ii, measurement_data_channel in enumerate(measurement_data):
            data_processed = np.array(measurement_data_channel)

            if videomode.diff_dir is not None:
                if isinstance(videomode.diff_dir, (list, tuple)):
                    diff_dir = videomode.diff_dir[ii]
                else:
                    diff_dir = videomode.diff_dir
                data_processed = qtt.utilities.tools.diffImageSmooth(
                    data_processed, dy=diff_dir, sigma=diff_sigma)

            if videomode.smoothing:
                data_processed = qtt.algorithms.generic.smoothImage(
                    data_processed)

            if videomode.laplace:
                data_processed = ndimage.laplace(
                    data_processed, mode='nearest')
            dd.append(data_processed)
        return dd

    @classmethod
    def plot_title(self, index):
        """ Return title for plot window """
        plot_title = f'plot {index}'
        return plot_title

    def create_dataset(self, processed_data, metadata):
        alldata = [None] * len(processed_data)
        for jj, data_block in enumerate(processed_data):
            if self.scan_dimension() == 1:
                xdata = np.arange(0., data_block.size)
                alldatax = qtt.data.makeDataSet1Dplain(
                    xname='x', x=xdata, yname='signal', y=data_block, loc_record={'label': 'videomode_1d_single'})
            else:
                xdata = np.arange(0., data_block.shape[1])
                ydata = np.arange(0., data_block.shape[0])
                alldatax = qtt.data.makeDataSet2Dplain(
                    xname='x', x=xdata,
                    yname='y', y=ydata,
                    zname='signal', z=data_block,
                    loc_record={'label': 'videomode_1d_single'})
            alldatax.metadata = copy.copy(metadata)
            alldata[jj] = alldatax
        return alldata

    def acquisition_device_type(self):
        """ Return type of acquisition device

        Returns:
            Device type as a string. Can be
        """
        measurement_instrument_handle = self.minstrumenthandle

        if isinstance(measurement_instrument_handle, AcquisitionScopeInterface):
            return 'AcquisitionScopeInterface'
        elif measurement_instrument_handle.name in ['digitizer', 'm4i']:
            return 'm4i'
        elif measurement_instrument_handle.name in ['ZIUHFLI', 'ziuhfli']:
            return 'ziuhfli'
        else:
            return 'other'


class DummyVideoModeProcessor(VideoModeProcessor):

    def __init__(self, station, verbose=1):
        """ Dummy implementation of the VideoModeProcessor """
        self.station = station

    def ppt_notes(self):
        return f'dummy notes for {self}'

    def initialize(self, videomode):
        pass

    def stop(self):
        pass

    def measure(self, videomode):
        data = [np.random.rand(100, )]

        return data

    def process(self, measurement_data, videomode):
        dd = self.default_processing(measurement_data, videomode)
        return dd

    def scan_dimension(self):
        return 1


class VideomodeSawtoothMeasurement(VideoModeProcessor):

    def __init__(self, station, verbose=1):
        self.station = station
        self.verbose = verbose

        self.scan_parameters = {}
        self.period_1d = qcodes.ManualParameter('period_1d', initial_value=1e-3)

    def plot_title(self, index):
        plot_title = str(self.minstrumenthandle) + ' ' + str(self.channels[index])
        return plot_title

    def create_dataset(self, processed_data, metadata):
        alldata = [None] * len(processed_data)
        if processed_data.ndim == 2:
            for jj, data_block in enumerate(processed_data):
                dataset, _ = makeDataset_sweep(data_block, self.sweepparams, self.sweepranges[0],
                                               gates=self.station.gates, loc_record={'label': 'videomode_1d'})
                dataset.metadata = copy.copy(metadata)
                alldata[jj] = dataset
        elif processed_data.ndim == 3:
            for jj, data_block in enumerate(processed_data):
                dataset, _ = makeDataset_sweep_2D(data_block, self.station.gates, self.sweepparams,
                                                  self.sweepranges,
                                                  loc_record={'label': 'videomode_2d'})
                dataset.metadata = copy.copy(metadata)
                alldata[jj] = dataset
        else:
            raise Exception('makeDataset: data.ndim %d' % processed_data.ndim)

        return alldata

    def set_properties(self, waveform, measurement_instrument, resolution=None):
        """ Create callback object for videmode data

        Args:
            station (QCoDeS station)
            waveform
            measurement_instrument (tuple): instrumentname, channel
            diff_dir (list or (int, str)): differentiation modes for the data
        """
        self.waveform = waveform
        self.minstrument = measurement_instrument[0]
        self.channels = measurement_instrument[1]
        self.measuresegment_arguments = {'mV_range': 5000}

        if not isinstance(self.channels, list):
            self.channels = [self.channels]

        self.unique_channels = list(np.unique(self.channels))

        self.resolution = resolution

    def update_position(self, position, verbose=1):
        if verbose:
            print('# %s: update position: %s' % (self.__class__.__name__, position,))

        station = self.station
        if self.scan_dimension() == 1:
            delta = position[0]
            if isinstance(self.scan_parameters['sweepparams'], str):
                param = getattr(station.gates, self.scan_parameters['sweepparams'])
                param.set(delta)
                if verbose > 2:
                    print('  set %s to %s' % (param, delta))
                return
        try:
            for ii, parameter in enumerate(self.scan_parameters['sweepparams']):
                delta = position[ii]
                if verbose:
                    print('param %s: delta %.3f' % (parameter, delta))

                if isinstance(parameter, str):
                    if parameter == 'gates_horz' or parameter == 'gates_vert':
                        d = self.scan_parameters['sweepparams'][parameter]
                        for gate, factor in d.items():
                            if verbose >= 2:
                                print('  %s: increment %s with %s' % (parameter, gate, factor * delta))
                            param = getattr(station.gates, gate)
                            param.increment(factor * delta)
                    else:
                        if verbose > 2:
                            print('  set %s to %s' % (parameter, delta))
                        param = getattr(station.gates, parameter)
                        param.set(delta)
                else:
                    raise Exception('_update_position with parameter type %s not supported' % (type(parameter),))
        except Exception as ex:
            logging.exception(ex)

    def ppt_notes(self):
        return str(self.scan_parameters)

    def initialize(self, videomode):
        scan_dimensions = self.scan_dimension()
        virtual_awg = getattr(self.station, 'virtual_awg', None)
        awg = getattr(self.station, 'awg', None)

        if isinstance(self.minstrumenthandle, AcquisitionScopeInterface):
            if scan_dimensions == 1:
                self.minstrumenthandle.period = self.period_1d()
            elif scan_dimensions == 2:
                self.minstrumenthandle.number_of_samples = np.prod(self.resolution)
            scan_channels = self.scanparams['minstrument'][1]
            channel_attributes = self.scanparams['minstrument'][2]
            self.minstrumenthandle.enabled_channels = tuple(scan_channels)
            for channel, attribute in zip(scan_channels, channel_attributes):
                self.minstrumenthandle.set_input_signal(channel, attribute)
            self.minstrumenthandle.start_acquisition()

        if scan_dimensions == 1:
            self._run_1d_scan(awg, virtual_awg, period=self.period_1d())
        elif scan_dimensions == 2:
            self._run_2d_scan(awg, virtual_awg)
        else:
            raise Exception(f'type of scan not supported scan_dimensions is {scan_dimensions}')

    def stop(self):
        if self.station is not None:
            if hasattr(self.station, 'awg'):
                self.station.awg.stop()
            if hasattr(self.station, 'RF'):
                self.station.RF.off()
            if hasattr(self.station, 'virtual_awg'):
                self.station.virtual_awg.stop()
                self.station.virtual_awg.disable_outputs()
        if isinstance(self.minstrumenthandle, AcquisitionScopeInterface):
            self.minstrumenthandle.stop_acquisition()

    def measure(self, videomode):

        if self.acquisition_device_type() == 'm4i':
            if self.scan_dimension() == 1:
                if (self.sampling_frequency() * self.period_1d()) * (1 - self.waveform['width']) / 2 > 40 + 2 * 16:
                    trigger_re_arm_compensation = True
                else:
                    trigger_re_arm_compensation = False
            else:
                trigger_re_arm_compensation = True

            trigger_re_arm_compensation = getattr(self, 'trigger_re_arm_compensation', trigger_re_arm_compensation)
            device_parameters = {'trigger_re_arm_compensation': trigger_re_arm_compensation}
        else:
            device_parameters = {}
        self._device_parameters = device_parameters
        data = qtt.measurements.scans.measuresegment(
            self.waveform, videomode.Naverage(), self.minstrumenthandle, self.unique_channels,
            **self.measuresegment_arguments, device_parameters=device_parameters)
        if np.all(data == 0):
            raise Exception('data returned contained only zeros, aborting')

        return data

    def process(self, measurement_data, videomode):
        measurement_data_selected = []
        for _, channel in enumerate(self.channels):
            unique_channel_idx = self.unique_channels.index(channel)
            data_processed = np.array(measurement_data[unique_channel_idx])
            measurement_data_selected.append(data_processed)

        dd = self.default_processing(measurement_data_selected, videomode)
        return dd

    def parse_instrument(self, measurement_instrument_handle, sample_rate):
        measurement_instrument_handle = qtt.measurements.scans.get_instrument(measurement_instrument_handle,
                                                                              self.station)
        self.minstrumenthandle = measurement_instrument_handle

        device_type = self.acquisition_device_type()

        if device_type == 'AcquisitionScopeInterface':
            if sample_rate != 'default':
                measurement_instrument_handle.sample_rate = sample_rate
            self.sampling_frequency = lambda: measurement_instrument_handle.sample_rate
        elif device_type == 'm4i':
            if sample_rate == 'default':
                self.sampling_frequency = measurement_instrument_handle.sample_rate
            else:
                measurement_instrument_handle.sample_rate(sample_rate)
                self.sampling_frequency = measurement_instrument_handle.sample_rate
        elif device_type == 'ziuhfli':
            self.sampling_frequency = measurement_instrument_handle.scope_samplingrate
        else:
            try:
                measurement_instrument_handle = qtt.measurements.scans.get_instrument(
                    measurement_instrument_handle, self.station)
                self.sampling_frequency = measurement_instrument_handle.sample_rate
            except Exception as ex:
                raise Exception(
                    f'no fpga or digitizer found minstrumenthandle is {measurement_instrument_handle}') from ex
        return self.sampling_frequency

    def _run_1d_scan(self, awg, virtual_awg, period=1e-3):
        if virtual_awg:
            if not isinstance(self.sweepparams, (str, dict)):
                raise Exception('argument sweepparams of type %s not supported' % type(self.sweepparams))
            sweep_range = self.sweepranges[0]
            gates = self.sweepparams if isinstance(self.sweepparams, dict) else {self.sweepparams: 1}
            waveform = virtual_awg.sweep_gates(gates, sweep_range, period)
            virtual_awg.enable_outputs(list(gates.keys()))
            virtual_awg.run()
        else:
            if isinstance(self.sweepparams, str):
                waveform, _ = awg.sweep_gate(self.sweepparams, self.sweepranges[0], period=period)
            elif isinstance(self.sweepparams, dict):
                waveform, _ = awg.sweep_gate_virt(self.sweepparams, self.sweepranges[0], period=period)
            else:
                raise Exception('argument sweepparams of type %s not supported' % type(self.sweepparams))
        self.set_properties(waveform, measurement_instrument=(self.minstrumenthandle, self.scan_parameters['channels']))

    def _run_2d_scan(self, awg, virtual_awg, period=None):
        if virtual_awg:
            sweep_ranges = [i for i in self.sweepranges]
            if isinstance(self.sweepparams[0], dict):
                gates = self.sweepparams
            else:
                # convert list of strings to dict format
                gates = self.sweepparams
                gates = [dict([(gate, 1)]) for gate in gates]

            if period is None:
                sampling_frequency = qtt.measurements.scans.get_sampling_frequency(self.minstrumenthandle)
                base_period = 1. / sampling_frequency
                total_period = base_period * np.prod(self.resolution)
            else:
                total_period = period
                warnings.warn('code untested')
            waveform = virtual_awg.sweep_gates_2d(gates, sweep_ranges, total_period, self.resolution)
            keys = qtt.utilities.tools.flatten([list(item.keys()) for item in gates])
            virtual_awg.enable_outputs(keys)
            virtual_awg.run()
        else:
            if isinstance(self.sweepparams, list):
                waveform, _ = awg.sweep_2D(self.sampling_frequency.get(), self.sweepparams,
                                           self.sweepranges, self.resolution)
            elif isinstance(self.sweepparams, dict):
                waveform, _ = awg.sweep_2D_virt(self.sampling_frequency.get(), self.sweepparams[
                    'gates_horz'], self.sweepparams['gates_vert'], self.sweepranges, self.resolution)
            else:
                raise Exception('argument sweepparams of type %s not supported' % type(self.sweepparams))
            if self.verbose:
                print('%s: 2d scan, define callback ' % (self.__class__.__name__,))

        self.set_properties(waveform, measurement_instrument=(self.minstrumenthandle, self.scan_parameters['channels']),
                            resolution=self.resolution)

    def set_scan_parameters(self, scan_parameters):
        self.scan_parameters = scan_parameters
        self.sweepranges = scan_parameters['sweepranges']

        if isinstance(self.sweepranges, numbers.Number):
            self.sweepranges = [self.sweepranges]
        self.sweepparams = scan_parameters['sweepparams']
        self.resolution = scan_parameters['resolution']
        self.channels = scan_parameters['channels']

    def scan_dimension(self):
        if isinstance(self.sweepranges, (float, int)):
            return 1
        elif isinstance(self.sweepranges, list):
            if len(self.sweepranges) == 2:
                return 2
            elif len(self.sweepranges) == 1:
                return 1
            else:
                raise Exception('scan dimension not supported')
        else:
            return -1

    def extend_videomode_name(self, name):
        """ String to append to VideoMode name """
        if name is not None:
            if isinstance(self.sweepparams, (str, list)):
                name += ': %s' % str(self.sweepparams)
        return name
