Algorithms
============

Algorithms have as input data (typically as a QCoDeS dataset) and parameters of the algorithm. 
The output is a dictionary with the main results as keys of the dictionary.


Quantum dots
------------

* :doc:`algorithms/fermidirac`
* Fit anti-crossing
* Virtual gate matrix
* Fit polarization line
* ...

Qubits
------
Why is this here? What do we want to say?

* PSB
* T1
* ...

Example notebooks
-----------------

.. toctree::
    :maxdepth: 3
    :titlesonly:
    :glob:

Core examples

* [Simple measurements](example_simple.ipynb) Simple measurement example
* [Defining a station](example_station.ipynb) How to define a station (e.g. your hardware setup)
* [Parameter widget](example_param_widget.py) Create a GUI to view parameters of an instrument
* [example_anticrossing.ipynb](example_anticrossing.ipynb) Automatically fit an anti-crossing in a dataset
* [Measurement control](example_measurement_control.py) Open a GUI to abort measurements

More advanced topics

* `[example_videomode.ipynb] <example_videomode.ipynb>` Videomode tuning
* [Virtual gate space]() TODO
* [example_coulomb_peak.ipynb](example_coulomb_peak.ipynb) Fitting a Coulomb peak
* [example_fermi_fitting.ipynb](example_fermi_fitting.ipynb) Automatically fit a Fermi distribution
* [example_polFitting.ipynb](example_polFitting.ipynb) Automatically fit a polarization line
* [example_charge_sensor.ipynb](example_charge_sensor.ipynb) Make corrections for the non-linearity of a charge sensor 
* [example_classical_dot_simulation.ipynb](example_classical_dot_simulation.ipynb) Simulate quantum dot systems	
	
    notebooks/example_simple.ipynb
    notebooks/example_station.ipynb
    notebooks/example_ohmic.ipynb
    notebooks/example_sensingdottuning.ipynb
#    notebooks/example_*


Back to the :doc:`main page <index>`.



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
