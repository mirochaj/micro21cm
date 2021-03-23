"""

analysis.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm
from .util import labels, bin_e2c, bin_c2e, get_error_2d

_default_modes = np.logspace(-1, 0., 21)
_default_colors = ['k', 'b', 'm', 'c', 'r', 'g', 'y', 'orange']
_default_ls = ['-', '--', '-.', ':']

def read_mcmc():
    pass

def plot_triangle(flatchain, fig=1, elements=[0,1], complement=False,
    bins=20, burn=0, fig_kwargs={}, contours=True, fill=False,
    nu=[0.95, 0.68], take_log=False, is_log=False, labels=None,
    **kwargs):
    """

    """

    fig = pl.figure(constrained_layout=True, num=fig, **fig_kwargs)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    Np = len(elements)
    gs = fig.add_gridspec(Np, Np)

    if type(bins) not in [list, tuple, np.ndarray]:
        bins = [bins] * Np
    if type(complement) not in [list, tuple, np.ndarray]:
        complement = [complement] * Np
    if type(is_log) not in [list, tuple, np.ndarray]:
        is_log = [is_log] * Np
    if type(take_log) not in [list, tuple, np.ndarray]:
        take_log = [take_log] * Np

    # Remember, for gridspec, rows are numbered frop top-down.
    axes_by_row = [[] for i in range(Np)]
    for i, row in enumerate(range(Np)):
        for j, col in enumerate(range(Np)):
            # Skip elements in upper triangle
            if j > i:
                continue

            # Create axis
            _ax = fig.add_subplot(gs[i,j])
            axes_by_row[i].append(_ax)

            # Retrieve data to be used in plot
            if not is_log[i]:
                p1 = 1. - flatchain[burn:,elements[i]] if complement[i] \
                    else flatchain[burn:,elements[i]]
            else:
                p1 = 10**flatchain[burn:,elements[i]] if is_log[i] \
                    else flatchain[burn:,elements[i]]

            if take_log[i]:
                p1 = np.log10(p1)

            # 2-D PDFs from here on
            if not is_log[j]:
                p2 = 1. - flatchain[burn:,elements[j]] if complement[j] \
                    else flatchain[burn:,elements[j]]
            else:
                p2 = 10**flatchain[burn:,elements[j]] if is_log[j] \
                    else flatchain[burn:,elements[j]]

            if take_log[j]:
                p2 = np.log10(p2)

            # 1-D PDFs
            if i == j:
                _ax.hist(p2, density=True, bins=bins[j])
                if j > 0:
                    _ax.set_yticklabels([])
                    if j == Np - 1:
                        _ax.set_xlabel(labels[j])
                    else:
                        _ax.set_xticklabels([])
                else:
                    _ax.set_ylabel(r'PDF')

                _ax.set_xlim(p2.min(), p2.max())
                continue

            if contours:
                hist, be2, be1 = np.histogram2d(p2, p1, [bins[j], bins[i]])
                bc1 = bin_e2c(be1)
                bc2 = bin_e2c(be2)

                nu, levels = get_error_2d(p2, p1, hist, [bc2, bc1], nu=nu)

                # (columns, rows, histogram)
                if fill:
                    _ax.contourf(bc2, bc1, hist.T / hist.max(),
                        levels, zorder=4, **kwargs)
                else:
                    _ax.contour(bc2, bc1, hist.T / hist.max(),
                        levels, zorder=4, **kwargs)
            else:
                h, x, y, img = _ax.hist2d(p2, p1, bins=[bins[j], bins[i]],
                    cmap='viridis', norm=LogNorm())

            # Get rid of labels/ticks on interior panels.
            if i < Np - 1:
                _ax.set_xticklabels([])
            else:
                _ax.set_xlabel(labels[j])

            if j > 0:
                _ax.set_yticklabels([])
            else:
                _ax.set_ylabel(labels[i])

            _ax.set_ylim(p1.min(), p1.max())
            _ax.set_xlim(p2.min(), p2.max())

    # Done
    return fig, axes_by_row

def plot_ps(model=None, data=None, ax=None, fig=None, dimensionless=True,
    data_kwargs={}, model_kwargs={}, fig_kwargs={}, z=None, **kwargs):
    """
    Plot some power spectra and data.

    """

    # Data first, if supplied.
    if data is not None:
        data_keys = list(data.keys())
        if np.array(data_keys).dtype in [np.float64, np.int, np.float]:
            Nz = len(data_keys)
        else:
            Nz = 1
    else:
        Nz = 1

    # Setup plot window
    if ax is None:
        fig, axes = pl.subplots(1, Nz, **fig_kwargs)
        if Nz == 1:
            axes = [axes]
    else:
        axes = [ax]
        if fig is None:
            fig = None

    ##
    # Data first, if provided.
    if data is not None:
        for i, ax in enumerate(axes):
            if 'k' in data:
                _data = data
                _key = None
            else:
                _key = data_keys[i]
                _data = data[_key]

            k = _data['k']

            if 'D_sq' in _data:
                Dsq = _data['D_sq']
            else:
                print("WARNING: No data found.")
                continue

            err = _data['err']

            ax.errorbar(k, Dsq + 2 * err, yerr=err, **data_kwargs)

            if _key is not None:
                ax.set_title(r'$z=%.1f$' % _key)

    ##
    # On to the models
    if model is not None:
        # model_inst, model_kw, k, z = model

        if data is not None:
            if 'k' in data:
                redshifts = None
                assert 'z' in model_kwargs
            else:
                redshifts = data_keys
        else:
            redshifts = None

        kw = model_kwargs.copy()
        if 'k' not in model_kwargs:
            kw['k'] = _default_modes
        k = kw['k']
        kw_not_args = kw.copy()
        if 'z' in kw_not_args:
            del kw_not_args['z']
        del kw_not_args['k']

        for i, ax in enumerate(axes):
            if redshifts is not None:
                z = redshifts[i]
            else:
                z = model_kwargs['z']

            ps_21 = model.get_ps_21cm(z=z, k=k, **kw_not_args)
            Dsq = ps_21 * k **3 / 2. / np.pi**2
            axes[i].loglog(kw['k'], Dsq, **kwargs)

    ##
    # Clean-up axes
    for i, ax in enumerate(axes):
        ax.set_xlabel(labels['k'])
        ax.set_xscale('log')
        ax.set_yscale('log')

        if i > 0:
            continue

        ax.set_ylabel(labels['delta_sq_long'])

    return fig, axes


def plot_ps_multi(split_by, color_by, ls_by, z=None, model=None,
    data=None, axes=None, fig=None, dimensionless=True, data_kwargs={},
    model_kwargs={}, fig_kwargs={}, colors=None, ls=None, **kwargs):
    """

    """

    if data is not None:
        if 'k' not in data:
            assert z is not None
            data = data[z]
        else:
            assert 'k' in data, \
                "Must supply data one redshift at a time for this routine."

    if colors is None:
        colors = _default_colors
    if ls is None:
        ls = _default_ls

    svals = model_kwargs[split_by]
    cvals = model_kwargs[color_by]
    lvals = model_kwargs[ls_by]
    ncols = max(len(svals), 1)

    if axes is None:
        fig, axes = pl.subplots(1, ncols, **fig_kwargs)
        if ncols == 1:
            axes = [axes]


    # Data first
    for i, sval in enumerate(svals):
        plot_ps(model=None, data=data, z=z, ax=axes[i],
            data_kwargs=data_kwargs)

        for j, cval in enumerate(cvals):
            for k, lval in enumerate(lvals):
                kw = model_kwargs.copy()
                kw[split_by] = svals[i]
                kw[color_by] = cvals[j]
                kw[ls_by] = lvals[k]

                kw2 = kwargs.copy()
                kw2['color'] = colors[j]
                kw2['ls'] = ls[k]

                plot_ps(model=model, data=None, ax=axes[i],
                    model_kwargs=kw, **kw2)

        if i > 0:
            axes[i].set_ylabel('')

    return fig, axes

def plot_igm_constraints(sampler, model, burn=0, bins=20):
    chain = sampler.flatchain

    samples = {}
    for i, par in enumerate(model.astro_params):
        samples[par] = chain[burn:,i]

    fig = pl.figure(constrained_layout=True, figsize=(12, 8))
    grid = fig.add_gridspec(2,2)

    ax_xH = fig.add_subplot(grid[0, 0])
    ax_Ts = fig.add_subplot(grid[1, 1])
    ax_2d = fig.add_subplot(grid[1, 0])

    #bins = [np.arange(0., 1.025, 0.025), np.arange(0, 10.5, 0.5)]
    h, x, y, img = ax_2d.hist2d(1 - samples['Q'], samples['Ts'], bins=bins,
        cmap='viridis')


    ax_Ts.hist(samples['Ts'], density=True, bins=bins[1])
    ax_xH.hist(1-samples['Q'], density=True, bins=bins[0])

    ax_Ts.set_xlabel(labels['Ts'])

    #ax_2d.plot([0., 1], [Tgas_z8]*2, color='w', ls='--', lw=3)
    ax_2d.set_xlim(0, 1.0)
    ax_2d.set_ylim(0, 10)
    ax_xH.set_xlim(0, 1)

    ax_2d.set_xlabel(labels['xHI'])
    ax_2d.set_ylabel(labels['Ts'])
    #ax_2d.annotate(r'$T_S = T_{\mathrm{ad}}(z=7.9)$', (0, Tgas_z8),
    #       ha='left', va='bottom', color='w', fontsize=16)
    #ax_Ts.plot([Tgas_z8]*2, [0, 0.5], color='k', ls='--')

def plot_walker_trajectories(sampler, model):
    pass

def get_limits_on_params(sampler, model, percentile=(0.025, 0.975)):

    pcen_100 = np.array(percentile)*100

    results = {}
    for i, par in enumerate(model.params):
        lo, hi = np.percentile(sampler.flatchain[:,i], pcen_100)
        results[par] = lo, hi

    return results
