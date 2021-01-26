"""

analysis.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import numpy as np
from .util import labels
import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm

_default_modes = np.logspace(-1, 0., 21)
_default_colors = ['k', 'b', 'm', 'c', 'r', 'g', 'y', 'orange']
_default_ls = ['-', '--', '-.', ':']

def read_mcmc():
    pass

def plot_triangle(flatchain, fig=1, elements=[0,1], complement=[False, True],
    bins=20, burn=0, fig_kwargs={}, contours=True, fill=False, nu=[0.95, 0.68],
    **kwargs):
    """

    """

    fig = pl.figure(constrained_layout=True, num=fig, **fig_kwargs)
    gs = fig.add_gridspec(2,2)

    ax_p2 = fig.add_subplot(gs[0, 0])
    ax_p1 = fig.add_subplot(gs[1, 1])
    ax_2d = fig.add_subplot(gs[1, 0])

    p1 = flatchain[burn:,elements[0]]
    p2 = flatchain[burn:,elements[1]]

    if complement[0]:
        p1 = 1. - p1
    if complement[1]:
        p2 = 1. - p2

    if type(bins) not in [list, tuple, np.ndarray]:
        bins = [bins] * 2

    if contours:
        hist, be1, be2 = np.histogram2d(p1, p2, bins)
        bc1 = bin_e2c(be1)
        bc2 = bin_e2c(be2)

        nu, levels = get_error_2d(p1, p2, hist, [bc1, bc2], nu=nu)

        ax_2d.contour(bc2, bc1, hist / hist.max(),
            levels, zorder=4, **kwargs)
    else:
        h, x, y, img = ax_2d.hist2d(p2, p1, bins=bins[-1::-1], cmap='viridis')

    ax_p1.hist(p1, density=True, bins=bins[0])
    ax_p2.hist(p2, density=True, bins=bins[1])

    return fig, ax_p1, ax_p2, ax_2d

def get_error_2d(x, y, z, bins, nu=[0.95, 0.68], weights=None, method='raw'):
    """
    Find 2-D contour given discrete samples of posterior distribution.

    Parameters
    ----------
    x : np.ndarray
        Array of samples in x.
    y : np.ndarray
        Array of samples in y.
    bins : np.ndarray, (2, Nsamples)

    method : str
        'raw', 'nearest', 'linear', 'cubic'


    """

    if method == 'raw':
        nu, levels = _error_2D_crude(z, nu=nu)
    else:

        # Interpolate onto new grid
        grid_x, grid_y = np.meshgrid(bins[0], bins[1])
        points = np.array([x, y]).T
        values = z

        grid = griddata(points, z, (grid_x, grid_y), method=method)

        # Mask out garbage points
        mask = np.zeros_like(grid, dtype='bool')
        mask[np.isinf(grid)] = 1
        mask[np.isnan(grid)] = 1
        grid[mask] = 0

        nu, levels = _error_2D_crude(grid, nu=nu)

    return nu, levels

def _error_2D_crude(L, nu=[0.95, 0.68]):
    """
    Integrate outward at "constant water level" to determine proper
    2-D marginalized confidence regions.

    ..note:: This is fairly crude -- the "coarse-ness" of the resulting
        PDFs will depend a lot on the binning.

    Parameters
    ----------
    L : np.ndarray
        Grid of likelihoods.
    nu : float, list
        Confidence intervals of interest.

    Returns
    -------
    List of contour values (relative to maximum likelihood) corresponding
    to the confidence region bounds specified in the "nu" parameter,
    in order of decreasing nu.
    """

    if type(nu) in [int, float]:
        nu = np.array([nu])

    # Put nu-values in ascending order
    if not np.all(np.diff(nu) > 0):
        nu = nu[-1::-1]

    peak = float(L.max())
    tot = float(L.sum())

    # Counts per bin in descending order
    Ldesc = np.sort(L.ravel())[-1::-1]

    Lencl_prev = 0.0

    # Will correspond to whatever contour we're on
    j = 0

    # Some preliminaries
    contours = [1.0]
    Lencl_running = []

    # Iterate from high likelihood to low
    for i in range(1, Ldesc.size):

        # How much area (fractional) is contained in bins at or above the current level?
        Lencl_now = L[L >= Ldesc[i]].sum() / tot

        # Keep running list of enclosed (integrated) likelihoods
        Lencl_running.append(Lencl_now)

        # What contour are we on?
        Lnow = Ldesc[i]

        # Haven't hit next contour yet
        if Lencl_now < nu[j]:
            pass
        # Just passed a contour
        else:

            # Interpolate to find contour more precisely
            Linterp = np.interp(nu[j], [Lencl_prev, Lencl_now],
                [Ldesc[i-1], Ldesc[i]])

            # Save relative to peak
            contours.append(Linterp / peak)

            j += 1

            if j == len(nu):
                break

        Lencl_prev = Lencl_now

    # Return values that match up to inputs
    return nu[-1::-1], np.array(contours[-1::-1])

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

def bin_e2c(bins):
    """
    Convert bin edges to bin centers.
    """
    dx = np.diff(bins)
    assert np.allclose(np.diff(dx), 0), "Binning is non-uniform!"
    dx = dx[0]

    return 0.5 * (bins[1:] + bins[:-1])

def bin_c2e(bins):
    """
    Convert bin centers to bin edges.
    """
    dx = np.diff(bins)
    assert np.allclose(np.diff(dx), 0), "Binning is non-uniform!"
    dx = dx[0]

    return np.concatenate(([bins[0] - 0.5 * dx], bins + 0.5 * dx))
