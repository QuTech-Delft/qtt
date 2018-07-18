QuTech Tuning Cheat Sheet
============


Viewer from instrument parameters:
```
> qtt.createParameterWidget([gates])
```

Start measurement control unit:
```
> qtt.live_plotting.start_measurement_control()
```

Start data viewer:
```
import qtt.gui.dataviewer
dv=qtt.gui.dataviewer.DataViewer(datadir=r'P:\data')
```

Copy dataset to Powerpoint
```
qtt.tools.addPPT_dataset(data);
```
