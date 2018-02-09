Contributing
============

.. include::  ../CONTRIBUTING.md

Coding Style
NOTE(giulioungaretti): is this enough ?

* Comments should be for describing why you are doing something. If you feel you need a comment to explain what you are doing, the code could probably be rewritten more clearly.
* If you do need a multiline statement, use implicit continuation (inside parentheses or brackets) and implicit string literal concatenation rather than backslash continuation
* Docstrings are required for classes, attributes, methods, and functions (if public i.e no leading underscore). Because docstrings (and comments) are not code, pay special attention to them when modifying code: an incorrect comment or docstring is worse than none at all! Docstrings should utilize the google style in order to make them read well, regardless of whether they are viewed through help() or on Read the Docs. See the falcon framework for best practices examples.
* Use PEP8 style. Not only is this style good for readability in an absolute sense, but consistent styling helps us all read each other’s code.

