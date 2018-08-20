Contributing
============

We welcome all of you to contribute to QTT, your input is valuable so that we together can continue improving it. To keep 
the framework useable for everyone, we ask all contributors to keep to the guidelines laid out in this section. If there are issues you cannot solve yourself or if there are any other questions, 
contacting the main developers of QuTech Tuning can be done via `GitHub issues <https://github.com/VandersypenQutech/qtt/issues>`_. 

Code development
----------------

When contributing to QTT you will want to write a new piece of code. Please keep in mind the guidelines below to allow us to work together in an efficient way:

* Before starting contributing make a new branch on GitHub, where you can push your contributions to. You will not be able to push directly to the master branch.

* Make regular commits and clearly explain what you are doing.

* If you run into a problem you cannot solve yourself, please take up contact with our main developers via `GitHub issues <https://github.com/VandersypenQutech/qtt/issues>`_. 

* Always include a test function in the file of your function
	


Code style
----------

Because QTT is a framework used by many people, we need it to follow certain style guidelines. We try to follow the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ style guide, except that we allow lines to be up to 120 characters.
Many editors support `autopep8 <https://pypi.python.org/pypi/autopep8>`_ that can help with coding style. Below we list some basic coding style guidelines, including examples. Please follow them when contributing to QTT.
We also try to follow `PEP 20 <https://www.python.org/dev/peps/pep-0020/>`_.


* Docstrings are required for classes, attributes, methods, and functions (if public i.e no leading underscore).

* Document your functions before making a pull request into the QuTech Tuning main branch. An example of a well documented function is shown below:

  .. code:: python


		def _cost_double_gaussian(signal_amp, counts, params):
			""" Cost function for fitting of double Gaussian. 

			Args:
				signal_amp (array): x values of the data
				counts (array): y values of the data
				params (array): parameters of the two gaussians, [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up]
					amplitude of first (second) gaussian = A_dn (A_up) 
					standard deviation of first (second) gaussian = sigma_dn (sigma_up)
					average value of the first (second) gaussian = mean_dn (mean_up)

			Returns:
				cost (float): value which indicates the difference between the data and the fit

			"""
			model = double_gaussian(signal_amp, params)
			cost = np.linalg.norm(counts - model)

			return cost

					
* Since we are dealing with code in development:

   - For methods implementing an algorithm return a dictionary so that we can modify the output arguments without breaking backwards compatibility
   - Add arguments ``fig`` or ``verbose`` to function to provide flexible analysis and debugging


Uploading code
--------------

To upload code use git commit and git push. For the qtt repository always make a branch first. After
uploading a branch one can make a `pull request <https://help.github.com/articles/about-pull-requests/>`_ which will be reviewed for inclusion in QTT 
by our main developers. If the code is up to standards we will include it in the QTT repository.



Bugs reports and feature requests
---------------------------------

If you don't know how to solve a bug yourself or want to request a feature, you can raise an issue via github’s `issues <https://github.com/VandersypenQutech/qtt/issues>`_. Please first search for existing and closed issues, if your problem or idea is not yet addressed, please open a new issue.



