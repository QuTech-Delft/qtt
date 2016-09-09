import qcodes

#%% Static variables

mwindows = None
liveplotwindow = None


def livePlot():
    global liveplotwindow
    if liveplotwindow is not None:
        return liveplotwindow
    return None
