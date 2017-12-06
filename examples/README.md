# QuTech Tuning examples

Also see the [Qcodes examples](https://github.com/QCoDeS/Qcodes/tree/master/docs/examples)

## Core examples

* [Simple measurements](example_simple.ipynb) Simple measurement example
* [Defining a station](example_station.ipynb) How to define a station (e.g. your hardware setup)
* [Parameter widget](example_param_widget.py) Create a GUI to view parameters of an instrument
* [example_anticrossing.ipynb](example_anticrossing.ipynb) Automatically fit an anti-crossing in a dataset
* [Measurement control](example_measurement_control.py) Open a GUI to abort measurements

## More advanced topics

* [example_videomode.ipynb](example_videomode.ipynb) Videomode tuning
* [Virtual gate space]() TODO
* [example_coulomb_peak.ipynb](example_coulomb_peak.ipynb) Fitting a Coulomb peak
* [example_fermi_fitting.ipynb](example_fermi_fitting.ipynb) Automatically fit a Fermi distribution
* [example_polFitting.ipynb](example_polFitting.ipynb) Automatically fit a polarization line
* [example_charge_sensor.ipynb](example_charge_sensor.ipynb) Make corrections for the non-linearity of a charge sensor 
* [example_classical_dot_simulation.ipynb](example_classical_dot_simulation.ipynb) Simulate quantum dot systems

## Code snippets

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
