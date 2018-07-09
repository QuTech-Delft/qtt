Contributing
============

Contact: Pieter Eendebak pieter.eendebak@tno.nl

Code style
----------

* Docstrings are required for classes, attributes, methods, and functions (if public i.e no leading underscore).

.. code:: python


def plot_anticrossing(ds, afit, fig=100, linewidth=2):

    """ Plot fitted anti-crossing on dataset

    

    Args:

        afit (dict): fit data from fit_anticrossing

        ds (None or DataSet): dataset to show

        fig (int): index of matplotlib window

        linewidth: (integer) plot linewidth, default = 2



    Returns:

        -

    

    """

* Try to follow the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ style guide. Many editors support `autopep8 <https://pypi.python.org/pypi/autopep8>`_ that can help with coding style.
 ```
[pep8]
max-line-length = 120
```
* Since we are dealing with code in development:
   - For methods implementing an algorithm return a dictionary so that we can modify the output arguments without breaking backwards compatibility
   - Add arguments ``fig`` or ``verbose`` to function to provide flexible analysis and debugging

Uploading code
--------------

To upload code use git commit and git push. For the qtt repository always make a branch first. After
uploading a branch one can make a [pull request](https://help.github.com/articles/about-pull-requests/) which will be reviewed for inclusion in QTT.

For the `users` and `stations` repository you can upload directly to the master branch.

Bugs reports and feature requests
---------------------------------

This is what github’s `issues <https://github.com/VandersypenQutech/qtt/issues>`_ are for. Search for existing and closed issues. If your problem or idea is not yet addressed, please open a new issue.



