"""
A GUI for multi-processing logging using ZMQ

Code is adapted from https://github.com/zeromq/pyzmq/blob/master/examples/logger/zmqlogger.py

Pieter Eendebak <pieter.eendebak@tno.nl>

"""

# %% Import packages
import logging
import os
import signal
import time
import argparse
import re

from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import Signal, Slot

import zmq
import zmq.log.handlers
from zmq.log.handlers import PUBHandler

import pyqtgraph.multiprocess as mp
import qtt
# %% Util functions


def static_var(varname, value):
    """ Helper function to create a static variable """
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var("time", 0)
def tprint(string, dt=1, output=False):
    """ Print progress of a loop every dt seconds """
    if (time.time() - tprint.time) > dt:
        print(string)
        tprint.time = time.time()
        if output:
            return True
        else:
            return
    else:
        if output:
            return False
        else:
            return

# %% Functions for installing the logger


def removeZMQlogger(name=None, verbose=0):
    """ Remove ZMQ logger from handlers

    Args:
        name (str or logger)
        verbose (int)
    """
    if isinstance(name, str) or name is None:
        logger = logging.getLogger(name)
    else:
        logger = name

    for h in logger.handlers:
        if isinstance(h, zmq.log.handlers.PUBHandler):
            if verbose:
                print('removeZMQlogger: removing handler %s' % h)
            logger.removeHandler(h)


def installZMQlogger(port=5800, name=None, clear=True, level=None, logger=None):
    """ Add ZMQ logging handler to a Python logger 
    """

    if clear:
        removeZMQlogger(name)
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.RCVHWM, 10)

    pub.connect('tcp://127.0.0.1:%i' % port)

    if logger is None:
        logger = logging.getLogger()
    if level is not None:
        logger.setLevel(level)
    handler = PUBHandler(pub)
    pid = os.getpid()
    pstr = 'pid %d: ' % pid
    handler.formatters = {
        logging.DEBUG: logging.Formatter(pstr
                                         + "%(levelname)s %(filename)s:%(lineno)d - %(message)s\n"),
        logging.INFO: logging.Formatter(pstr + "%(message)s\n"),
        logging.WARN: logging.Formatter(pstr
                                        + "%(levelname)s %(filename)s:%(lineno)d - %(message)s\n"),
        logging.ERROR: logging.Formatter(pstr
                                         + "%(levelname)s %(filename)s:%(lineno)d - %(message)s - %(exc_info)s\n"),
        logging.CRITICAL: logging.Formatter(pstr +
                                            "%(levelname)s %(filename)s:%(lineno)d - %(message)s\n")}

    logger.addHandler(handler)
    logger.debug('installZMQlogger: handler installed')
    # first message always is discarded
    return logger

# %%


class zmqLoggingGUI(QtWidgets.QDialog):

    LOG_LEVELS = dict({logging.DEBUG: 'debug', logging.INFO: 'info',
                       logging.WARN: 'warning', logging.ERROR: 'error', logging.CRITICAL: 'critical'})

    def __init__(self, parent=None, extra_controls=False):
        """ Simple GUI to view log messages """
        super(zmqLoggingGUI, self).__init__(parent)

        self.setWindowTitle('ZMQ logger')

        self.imap = dict((v, k) for k, v in self.LOG_LEVELS.items())

        self._console = QtWidgets.QPlainTextEdit(self)
        self._console.setMaximumBlockCount(2000)

        self._button = QtWidgets.QPushButton(self)
        self._button.setText('Clear')

        self._levelBox = QtWidgets.QComboBox(self)
        for k in sorted(self.LOG_LEVELS.keys()):
            logging.debug('item %s' % k)
            val = self.LOG_LEVELS[k]
            self._levelBox.insertItem(k, val)

        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(self._button)
        if extra_controls:
            self._killbutton = QtWidgets.QPushButton(self)
            self._killbutton.setText('Kill processes')
            blayout.addWidget(self._killbutton)
            self._killbutton.clicked.connect(self.killPID)
        blayout.addWidget(self._levelBox)
        self._button.clicked.connect(self.clearMessages)
        self._levelBox.currentIndexChanged.connect(self.setLevel)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._console)
        layout.addLayout(blayout)
        self.setLayout(layout)

        self.addMessageSignal.connect(self._addMessage)
        self.addMessage('logging started...' + '\n')

        self._levelBox.setCurrentIndex(1)
        self.loglevel = logging.INFO
        self.nkill = 0

    def closeEvent(self, evnt):
        print('loggingGUI: close event')
        super().closeEvent(evnt)

    def setLevel(self, boxidx):
        name = self._levelBox.itemText(boxidx)
        lvl = self.imap.get(name, None)
        logging.debug('set level to %s: %d' % (name, lvl))
        if lvl is not None:
            self.loglevel = lvl

    addMessageSignal = Signal(str)

    @Slot(str)
    def _addMessage(self, msg):
        """ Helper function to solve threading issues """
        self._console.moveCursor(QtGui.QTextCursor.End)
        self._console.insertPlainText(msg)
        self._console.moveCursor(QtGui.QTextCursor.End)

    def addMessage(self, msg, level=None):
        """ Add a message to the GUI list """
        if level is not None:
            if level < self.loglevel:
                return
        self.addMessageSignal.emit(msg)

    def clearMessages(self):
        ''' Clear the messages in the logging window '''
        self._console.clear()
        self.addMessage('cleared messages...\n')
        self.nkill = 0

    def killPID(self):
        ''' Clear the messages in the logging window '''
        self.nkill = 10

    def setup_monitor(self, port=5800):
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.bind('tcp://127.0.0.1:%i' % port)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.setsockopt(zmq.RCVHWM, 10)

        # logging.basicConfig(level=level)
        app = QtWidgets.QApplication.instance()
        app.processEvents()

        logging.info('connected to port %s' % port)
        self.sub = sub

        from apscheduler.schedulers.background import BackgroundScheduler

        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        self.scheduler.add_job(self._callback, 'interval', seconds=1)

    def close(self):
        self.scheduler.pause()
        self.sub.close()

    def _callback(self, verbose=1):
        logging.debug('ZMQ logger: logging...')
        app = QtWidgets.QApplication.instance()
        dlg = self

        try:
            sub = self.sub
            for ij in range(10):
                # process at most 10 messages at a time
                level, message = sub.recv_multipart(zmq.NOBLOCK)
                # level, message = sub.recv_multipart()
                message = message.decode('ascii')
                if message.endswith('\n'):
                    # trim trailing newline, which will get appended again
                    message = message[:-1]
                level = level.lower().decode('ascii')
                log = getattr(logging, level)
                lvlvalue = dlg.imap.get(level, None)

                if verbose >= 2:
                    log(message)
                dlg.addMessage(message + '\n', lvlvalue)

                if dlg.nkill > 0:
                    print('check pid')
                    m = re.match(r'pid (\d*): heartbeat', message)
                    dlg.nkill = dlg.nkill - 1
                    if m is not None:
                        pid = int(m.group(1))
                        print('killing pid %d' % pid)
                        mysignal = getattr(signal, 'SIGKILL', signal.SIGTERM)
                        try:
                            os.kill(pid, mysignal)  # or signal.SIGKILL
                            dlg.addMessage(
                                'send kill signal to pid %d\n' % pid, logging.CRITICAL)
                        except Exception:
                            dlg.addMessage(
                                'kill signal to pid %d failed\n' % pid, logging.CRITICAL)
                            pass
            app.processEvents()

            if verbose >= 2:
                print('message: %s (level %s)' % (message, level))
        except zmq.error.Again:
            # no messages in system....
            app = QtWidgets.QApplication.instance()

            app.processEvents()
            time.sleep(.03)
            message = ''
            level = None
        if dlg.nkill > 0:
            time.sleep(.1)
            dlg.nkill = max(dlg.nkill - 1, 0)


def qt_logger(port, dlg, level=logging.INFO, verbose=1):
    raise Exception("do not use this function, use setup_monitor instead")


def start_logging_gui():
    """ Start logging GUI in the background """
    proc = mp.QtProcess()
    lp = proc._import('qtt.loggingGUI')
    mc = lp.zmqLoggingGUI()
    mc.show()
    mc.setup_monitor(port=5800)
    qtt._dummy_logging_gui = mc
    # return mc


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', default=1, help="verbosity level")
    parser.add_argument(
        '-l', '--level', default=logging.DEBUG, help="logging level")
    parser.add_argument('-p', '--port', type=int,
                        default=5800, help="zmq port")
    parser.add_argument('-g', '--gui', type=int, default=1, help="show gui")
    args = parser.parse_args()

    port = args.port
    verbose = args.verbose

    app = None
    if (not QtWidgets.QApplication.instance()):
        app = QtWidgets.QApplication([])
    dlg = zmqLoggingGUI()
    dlg.resize(800, 400)
    dlg.show()

    # start the log watcher
    try:
        dlg.setup_monitor(port)
    except KeyboardInterrupt:
        pass

    #dlg.close()

    def send_message_to_logger():
        port = 5800
        installZMQlogger(port=port, level=None)
        logging.warning('test')
