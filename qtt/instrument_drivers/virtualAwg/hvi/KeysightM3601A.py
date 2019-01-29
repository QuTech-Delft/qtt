import os
import logging

try:
    import keysightSD1
except ImportError:
    keysightSD1 = None

from qtt.instrument_drivers.virtualAwg.virtual_awg import VirtualAwg, VirtualAwgError


class KeysightM3601A(VirtualAwg):

    __awg_name = 'Keysight_M3201A'
    module_names = ['Module 0', 'Module 1', 'Module 2', 'Module 3']

    def __init__(self, awgs, settings, name='virtual_awg', logger=logging, **kwargs):
        super().__init__(awgs, settings, name, logger, **kwargs)
        if not keysightSD1:
            raise ImportError('To use the Keysight SD drivers install the keysightSD1 module '
                              '(http://www.keysight.com/main/software.jspx?ckey=2784055)')
        if not all(type(awg.fetch_awg).__name__ == KeysightM3601A.__awg_name for awg in self.awgs):
            raise VirtualAwgError('Unusable device added! Not a Keysight M3201A AWG.')
        self.hvi = keysightSD1.SD_HVI()
        self.load_hvi_file()
        self.set_hvi_modules()
        self.set_awg_settings()

    def set_hvi_modules(self):
        settings = zip(KeysightM3601A.module_names, self.awgs)
        for module, awg_wrapper in settings:
            error_code = self.hvi.assignHardwareWithUserNameAndModuleID(module, awg_wrapper.fetch_awg.awg)
            self.check_for_error(error_code)

    def load_hvi_file(self, hvi_file='sync_four_awgs.HVI'):
        file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../hvi/', hvi_file))
        error_code = self.hvi.open(file_path)
        self.check_for_error(error_code, show_warnings=False)

    def set_hvi_settings(self, length_sequence=100, number_of_repetitions=-1, steps=0):
        module_count = len(KeysightM3601A.module_names)
        for index in range(module_count):
            error_code = self.hvi.writeIntegerConstantWithIndex(index, "length_sequence", length_sequence)
            self.check_for_error(error_code)
            error_code = self.hvi.writeIntegerConstantWithIndex(index, "number_of_repetitions", number_of_repetitions)
            self.check_for_error(error_code)
            error_code = self.hvi.writeIntegerConstantWithIndex(index, "step", steps)
            self.check_for_error(error_code)

    def set_awg_settings(self):
        awg_count = len(self.awgs)
        for awg_number in range(awg_count):
            self.update_setting(awg_number, 'auto_trigger', keysightSD1.SD_TriggerModes.AUTOTRIG)

    def compile_hvi_file(self):
        error_code = self.hvi.compile()
        self.check_for_error(error_code)
        error_code = self.hvi.load()
        self.check_for_error(error_code)

    def check_for_error(self, error_code, show_warnings=True):
        if error_code == 0:
            return
        if error_code < 0:
            error_message = keysightSD1.SD_Error.getErrorMessage(error_code)
            raise VirtualAwgError('Error: (code {}) | {}'.format(error_code, error_message))
        if show_warnings and error_code > 0:
            error_message = keysightSD1.SD_Error.getErrorMessage(error_code)
            self._logger.warn('Warning: (code {}) | {}'.format(error_code, error_message))

    def run(self):
        super().run()
        self.hvi.start()

    def stop(self):
        self.hvi.stop()
        super().stop()

    def sequence_gates(self, sequences, do_upload=True):
        sweep_data = super().sequence_gates(sequences, do_upload)
        self.set_hvi_settings()
        self.compile_hvi_file()
        return sweep_data
