QuTech Tuning Cheat Sheet
============


Viewer from instrument parameters:

.. code-block:: console

	qtt.createParameterWidget([gates])


Start measurement control unit:

.. code-block:: console

	qtt.live_plotting.start_measurement_control()


Start data viewer:

.. code-block:: console

	import qtt.gui.dataviewer
	dv=qtt.gui.dataviewer.DataViewer(datadir=r'P:\data')


Copy dataset to Powerpoint

.. code-block:: console

	tt.tools.addPPT_dataset(data)
