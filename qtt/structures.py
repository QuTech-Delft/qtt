import numpy as np
import qcodes

from qtt.scans import scan1D
import qtt.data
from qtt.algorithms.coulomb import *
import matplotlib.pyplot as plt


class sensingdot_t:

    """ Class representing a sensing dot """

    def __init__(self, ggv, sdvalv, station=None, RFfreq=None, index=None, fpga_ch=None):
        self.verbose = 1
        self.gg = ggv
        self.sdval = sdvalv
        self.targetvalue = 800
        self.goodpeaks = None
        self.station = station
        self.RFfreq = RFfreq  # ?
        self.index = index
        self.instrument = 'keithley%d' % index
        if fpga_ch is None:
            self.fpga_ch = int(self.gg[1][2])
        else:
            self.fpga_ch = fpga_ch

        # values for measurement
        # RFfreq = None
        if index is not None:
            self.valuefunc = station.components[
                'keithley%d' % index].amplitude.get

    def __repr__(self):
        gates = self.station.gates
        s = 'sensingdot_t: %s: %s: g %.1f, value %.1f/%.1f' % (
            self.gg[1], str(self.sdval), gates.get(self.gg[1]), self.value(), self.targetvalue)
        # s='sensingdot_t: %s: %.1f '  % (self.gg[1], self.value() )
        return s

    def initialize(self, sdval=None, setPlunger=False):
        gates = self.station.gates
        if not sdval is None:
            self.sdval = sdval
        gg = self.gg
        for ii in [0, 2]:
            gates.set(gg[ii], self.sdval[ii])
        if setPlunger:
            ii = 1
            gates.set(gg[ii], self.sdval[ii])

    def tunegate(self):
        """ Return the gate used for tuning """
        return self.gg[1]

    def value(self):
        """ Return current through sensing dot """
        if self.valuefunc is not None:
            return self.valuefunc()
        global keithley1
        global keithley2
        if self.index == 1:
            keithley = keithley1
            kfactor = keithleyFactor(['keithley1'])[0]
        else:
            keithley = keithley2
            kfactor = keithleyFactor(['keithley2'])[0]
        if keithley is None:
            # raise Exception('sensingdot_t: keithley is None!')
            warnings.warn('keithley is None')
            return None
        Amp = 10
        readval = keithley.readnext()
        if readval is None:
            val = 0
        else:
            val = kfactor * readval * (1e12 / (Amp * 10e6))
        return val

    def scan1D(sd, outputdir=None, step=-2., max_wait_time=.75, scanrange=300):
        """ Make 1D-scan of the sensing dot """
        print('### sensing dot scan')
        keithleyidx = [sd.index]
        gg = sd.gg
        sdval = sd.sdval
        gates = sd.station.gates

        for ii in [0, 2]:
            gates.set(gg[ii], sdval[ii])

        startval = sdval[1] + scanrange
        startval = np.minimum(startval, 100)
        endval = sdval[1] - scanrange
        endval = np.maximum(endval, -700)

        scanjob1 = dict()
        scanjob1['sweepdata'] = dict(
            {'gates': [gg[1]], 'start': startval, 'end': endval, 'step': step})
        scanjob1['keithleyidx'] = keithleyidx
        scanjob1['compensateGates'] = []
        scanjob1['gate_values_corners'] = [[]]

        wait_time = sd.station.gate_settle(gg[1])
        wait_time = np.minimum(wait_time, max_wait_time)
        print('sensingdot_t: scan1D: gate %s, wait_time %.3f' %
              (sd.gg[1], wait_time))

        alldata = scan1D(
            scanjob1, sd.station, title_comment='plunger', wait_time=wait_time)

        # if not outputdir == None:
        #    saveCoulombData(outputdir, alldata)

        return alldata

    def scan2D(sd, ds=90, stepsize=-4, fig=None):
        """ Make 2D-scan of the sensing dot """
        keithleyidx = [index]
        gg = sd.gg
        sdval = sd.sdval

        set_gate(gg[1], sdval[1])

        scanjob = dict()
        scanjob['stepdata'] = dict(
            {'gates': [gg[0]], 'start': sdval[0] + ds, 'end': sdval[0] - ds, 'step': stepsize})
        scanjob['sweepdata'] = dict(
            {'gates': [gg[2]], 'start': sdval[2] + ds, 'end': sdval[2] - ds, 'step': stepsize})
        scanjob['keithleyidx'] = keithleyidx
        scanjob['compensateGates'] = []
        scanjob['gate_values_corners'] = [[]]

        alldata = scan2Djob(
            scanjob, TitleComment='2D', activegates=defaultactivegates(), wait_time=0.1)

        if fig is not None:
            show2D(alldata, fig=fig)
        return alldata

    def autoTune(sd, scanjob=None, fig=200, outputdir=None, correctdelay=True, step=-3., max_wait_time=.8, scanrange=300):
        if not scanjob is None:
            sd.autoTuneInit(scanjob)
        alldata = sd.scan1D(outputdir=outputdir, step=step,
                            scanrange=scanrange, max_wait_time=max_wait_time)

        x, y = qtt.data.dataset1Ddata(alldata)

        istep = float(np.abs(alldata.metadata['scanjob']['sweepdata']['step']))
        x, y = peakdataOrientation(x, y)

        goodpeaks = coulombPeaks(
            x, y, verbose=1, fig=fig, plothalf=True, istep=istep)
        if fig is not None:
            plt.title('autoTune: sd %d' % sd.index, fontsize=14)

        sd.goodpeaks = goodpeaks
        sd.tunex = x
        sd.tuney = y

        if len(goodpeaks) > 0:
            sd.sdval[1] = goodpeaks[0]['xhalfl']
            sd.targetvalue = goodpeaks[0]['yhalfl']
            # correction of gate delay
            if correctdelay:
                if sd.gg[1] == 'SD1b' or sd.gg[1] == 'SD2b':
                    # *(getwaittime(sd.gg[1])/max_wait_time )
                    corr = -step * .75
                    sd.sdval[1] += corr
        else:
            print('autoTune: could not find good peak')

        if sd.verbose:
            print(
                'sensingdot_t: autotune complete: value %.1f [mV]' % sd.sdval[1])
        return sd.sdval[1], alldata

    def autoTuneInit(sd, scanjob, mode='center'):
        stepdata = scanjob.get('stepdata', None)
        sweepdata = scanjob['sweepdata']
        # set sweep to center
        gates = sd.station.gates
        gates.set(
            sweepdata['gates'][0], (sweepdata['start'] + sweepdata['end']) / 2)
        if not stepdata is None:
            if mode == 'end':
                # set sweep to center
                gates.set(stepdata['gates'][0], (stepdata['end']))
            elif mode == 'start':
                # set sweep to center
                gates.set(stepdata['gates'][0], (stepdata['start']))
            else:
                # set sweep to center
                gates.set(
                    stepdata['gates'][0], (stepdata['start'] + stepdata['end']) / 2)

    def fineTune(sd, fig=300, stephalfmv=8):
        g = sd.tunegate()
        readfunc = sd.value

        if sd.verbose:
            print('fineTune: delta %.1f [mV]' % (stephalfmv))

        cvalstart = sd.sdval[1]
        sdstart = autotunePlunger(
            g, cvalstart, readfunc, dstep=.5, stephalfmv=stephalfmv, targetvalue=sd.targetvalue, fig=fig + 1)
        set_gate(g, sdstart)
        sd.sdval[1] = sdstart
        time.sleep(.5)
        if sd.verbose:
            print('fineTune: target %.1f, reached %.1f' %
                  (sd.targetvalue, sd.value()))
        return (sdstart, None)

    def autoTuneFine(sd, sweepdata=None, scanjob=None, fig=300):
        if sweepdata is None:
            sweepdata = scanjob['sweepdata']
            stepdata = scanjob.get('stepdata', None)
        g = sd.tunegate()
        gt = stepdata['gates'][0]
        cdata = stepdata
        factor = sdInfluenceFactor(sd.index, gt)
        d = factor * (cdata['start'] - cdata['end'])
        readfunc = sd.value

        if sd.verbose:
            print('autoTuneFine: factor %.2f, delta %.1f' % (factor, d))

        # set sweep to center
        set_gate(sweepdata['gates'][0], (sweepdata[
                 'start'] + sweepdata['end']) / 2)

        sdmiddle = sd.sdval[1]
        if 1:
            set_gate(cdata['gates'][0], (cdata['start'] + cdata['end']) / 2)

            sdmiddle = autotunePlunger(
                g, sd.sdval[1], readfunc, targetvalue=sd.targetvalue, fig=fig)

        set_gate(cdata['gates'][0], cdata['start'])  # set step to start value
        cvalstart = sdmiddle - d / 2
        sdstart = autotunePlunger(
            g, cvalstart, readfunc, targetvalue=sd.targetvalue, fig=fig + 1)
        if sd.verbose >= 2:
            print(' autoTuneFine: cvalstart %.1f, sdstart %.1f' %
                  (cvalstart, sdstart))

        set_gate(cdata['gates'][0], cdata['end'])  # set step to end value
        # cvalstart2=((sdstart+d) + (sd.sdval[1]+d/2) )/2
        cvalstart2 = sdmiddle + (sdmiddle - sdstart)
        if sd.verbose >= 2:
            print(' autoTuneFine: cvalstart2 %.1f = %.1f + %.1f (d %.1f)' %
                  (cvalstart2, sdmiddle, (sdmiddle - sdstart), d))
        sdend = autotunePlunger(
            g, cvalstart2, readfunc, targetvalue=sd.targetvalue, fig=fig + 2)

        return (sdstart, sdend, sdmiddle)

    def fastTune(self, Naverage=50, sweeprange=79, period=.5e-3, fig=201, sleeptime=2, delete=True):
        ''' Fast tuning of the sensing dot plunger '''

        waveform, sweep_info = self.station.awg.sweep_gate(
            self.gg[1], sweeprange, period, wave_name='fastTune_%s' % self.gg[1], delete=delete)

        qtt.time.sleep(sleeptime)

        ReadDevice = ['FPGA_ch%d' % self.fpga_ch]
        _, DataRead_ch1, DataRead_ch2 = self.station.fpga.readFPGA(
            Naverage=Naverage, ReadDevice=ReadDevice)

        self.station.awg.stop()

        if self.fpga_ch == 1:
            datr = DataRead_ch1
        else:
            datr = DataRead_ch2
        data = self.station.awg.sweep_process(datr, waveform, Naverage)

        sdplg = getattr(self.station.gates, self.gg[1])
        initval = sdplg.get()
        sweepvalues = sdplg[initval - sweeprange /
                            2:sweeprange / 2 + initval:sweeprange / len(data)]

<<<<<<< 351ec71d6c346302e59204b766a1af4f02c6225a
        alldata = qtt.data.makeDataSet1D(sweepvalues, preset_data=data)
=======
        alldata = qtt.data.makeDataSet1D(sweepvalues, y=data)
>>>>>>> Add work on scan2dfast

        y = np.array(alldata.arrays['measured'])
        x = alldata.arrays[self.gg[1]]
        x, y = peakdataOrientation(x, y)

        goodpeaks = coulombPeaks(
            x, y, verbose=1, fig=fig, plothalf=True, istep=1)

        if fig is not None:
            plt.title('autoTune: sd %d' % self.index, fontsize=14)

        if len(goodpeaks) > 0:
            self.sdval[1] = goodpeaks[0]['xhalfl']
            self.targetvalue = goodpeaks[0]['yhalfl']
        else:
            print('autoTune: could not find good peak, may need to adjust mirrorfactor')

        if self.verbose:
            print(
                'sensingdot_t: autotune complete: value %.1f [mV]' % self.sdval[1])

        return self.sdval[1], alldata
