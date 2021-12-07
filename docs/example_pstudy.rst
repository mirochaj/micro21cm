:orphan:

Simple Parameter Study
======================
A typical *micro21cm* model has four parameters: `Q`, `Ts`, `R`, and either `sigma` or `gamma`, depending on the choice of bubble size distribution (BSD). By default, a log-normal BSD is used, in which case `sigma` is the fourth free parameter. These represent:

* `Q`: the volume filling fraction of ionized bubbles.
* `Ts`: the spin temperature of gas in the neutral ``bulk'' intergalactic medium (IGM).
* `R`: the characteristic size of bubbles.
* `sigma`: the log-normal dispersion of a log-normal BSD.

A simple parameter study can be conducted via, e.g.,

::

    import micro21cm
    import numpy as np
    import matplotlib.pyplot as pl

    model = micro21cm.BubbleModel()


    # Set modes of interest
    k = np.logspace(-2., 0, 41)

    fig, ax = pl.subplots(1, 1, num=1)

    for i, Q in enumerate([0, 0.1, 0.2, 0.3]):
        ps = model.get_ps_21cm(z=8, k=k, Q=Q, R=3.)
        ax.loglog(k, k**3 * ps / 2. / np.pi**2, label=r'$Q={:.1f}$'.format(Q))

    ax.set_xlabel(micro21cm.util.labels['k'])
    ax.set_ylabel(micro21cm.util.labels['delta_sq'])
    ax.legend(loc='upper left')

Alternatively, you can vary the spin temperature,

::

    fig, ax = pl.subplots(1, 1, num=2)

    for i, Ts in enumerate([2, 5, 10, 100]):
        ps = model.get_ps_21cm(z=8, k=k, Q=0.3, Ts=Ts, R=3.)
        ax.loglog(k, k**3 * ps / 2. / np.pi**2, label=r'$T_S={:.1f}$'.format(Ts))

    ax.set_xlabel(micro21cm.util.labels['k'])
    ax.set_ylabel(micro21cm.util.labels['delta_sq'])
    ax.legend(loc='upper left')

By default, :math:`\sigma=0.5`, but one can vary that as well via the `sigma` keyword argument.
