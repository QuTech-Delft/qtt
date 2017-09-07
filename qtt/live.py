import qcodes

#%% Static variables

mwindows = None
liveplotwindow = None

def livePlot():
    """ Return handle to live plotting window """
    global liveplotwindow
    if liveplotwindow is not None:
        return liveplotwindow
    return None
