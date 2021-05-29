================
**micro21cm**
================

The code was written to provide a very simple phenomenological description of the 21-cm brightness temperature field during reionization. The goal is to proceed in a way that abstracts away assumptions about how galaxies form and how reionization proceeds to the maximum possible extent, and in so doing build intuition for how 21-cm fluctuations encode properties of the IGM, and determine the extent to which the priors of physically-motivated astrophysical models influence inferences drawn from high-:math:`z` 21-cm data.

The name of the code, *micro21cm*, is short for Model-Independent Constraints on Reionization from Observations of the 21-cm background. To be fair, the model-independent designation is debatable: the model assumes that reionization can be described by a patchwork of spherical bubbles whose abundance is well described by a simple distribution function. However, the code does not make any explicit assumptions about *astrophysical* models, e.g., how efficiently galaxies form stars.

The details of the model are described in Mirocha et al. (2021).

Dependencies
============
*micro21cm* depends on:

* `numpy <http://numpy.scipy.org/>`_
* `scipy <http://www.scipy.org/>`_
* `matplotlib <http://matplotlib.sourceforge.net>`_
* `camb <https://camb.readthedocs.io/en/latest/>`_

and optionally:

* `emcee <http://dan.iel.fm/emcee/current/>`_
* `powerbox <https://powerbox.readthedocs.io/en/latest/>`_
* `progressbar2 <http://progressbar-2.readthedocs.io/en/latest/>`_

Quick-Start
-----------
To make sure everything is working, a quick test is to generate a
realization of the 21-cm power spectrum using all default parameter values:

::

    import micro21cm
    import numpy as np
    import matplotlib.pyplot as pl

    # Initialize an object that does the heavy lifting
    model = micro21cm.BubbleModel()

    # Set modes of interest
    k = np.logspace(-2., 0, 41)

    # Compute P(k) at z=8 for a cold IGM 20% through reionization
    ps = model.get_ps_21cm(z=8, k=k, Q=0.2, R=3.)

    # Plot dimensionless power spectrum
    pl.loglog(k, k**3 * ps / 2. / np.pi**2)
    pl.xlabel(micro21cm.util.labels['k'])
    pl.ylabel(micro21cm.util.labels['delta_sq'])

Note that the bulk of the compute time is spent (i) computing the matter power spectrum through CAMB, which will be cached and re-used for subsequent calls to `get_ps_21cm`, and (ii) Fourier transforming the 21-cm correlation function to obtain the power spectrum. With a cached matter PS in hand, each 21-cm PS call should take only :math:`\sim 0.1` sec per :math:`k` mode.

More details in the pages listed below.

Contents
--------
.. toctree::
    :maxdepth: 1

    Home <self>
    example_pstudy
    example_fit
    example_internals
    example_params
