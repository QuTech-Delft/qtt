""" 
Contains functions to generate reports of scanned data
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings

try:
    from urllib import pathname2url as pathname2url
except:
    from urllib.request import pathname2url as pathname2url

import qtt.utilities.markup as markup
from qtt import pgeometry

warnings.warn('please do not import this module, it will be removed in the future', DeprecationWarning)

#%%


def reportTemplate(title):
    """ Create a markup object for a HTML page """
    page = markup.page()
    page.init(title=title,
              css=(), lang='en', bodyattrs=dict({'style': 'padding-left: 3px;'}),
              header="<h1>%s</h1><br/><span><a href=\"mailto:pieter.eendebak@tno.nl\">pieter.eendebak@tno.nl</a></span>" % title, metainfo=({'keywords': 'qutech, quantum dot, tuning, TNO'}),
              footer="<!-- End of page >")
    return page

#%%

