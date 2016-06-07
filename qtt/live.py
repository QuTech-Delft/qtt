import qcodes

#%% Static variables

mwindows = None
#liveplotwindow = None

def livePlot():
    global mwindows
    if mwindows is not None:
        return mwindows.get('plotwindow', None)
    return None
