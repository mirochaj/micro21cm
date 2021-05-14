"""

analysis.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import pickle
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage.filters import gaussian_filter
from .inference import tanh_generic, power_law, power_law_max1, \
    broken_power_law, broken_power_law_max1, extract_params, _all_pars
from .util import labels, bin_e2c, bin_c2e, get_error_2d

_default_modes = np.logspace(-1, 0., 21)
_default_colors = ['k', 'b', 'm', 'c', 'r', 'g', 'y', 'orange']
_default_ls = ['-', '--', '-.', ':']
_default_labels = {'Q': r'$Q$', 'R': r'$R$', 'Ts': r'$T_S$',
    'sigma': r'$\sigma$'}
_default_limits = {'Q': (-0.05, 1.05), 'R': (0, 15), 'Ts': (0, 150),
    'sigma': (0, 1)}
_default_z = np.arange(5, 20, 0.05)


bbox = dict(facecolor='none', edgecolor='k', fc='w',
        boxstyle='round,pad=0.3', alpha=0.9, zorder=1000)


class AnalyzeFit(object):
    def __init__(self, fn):
        self.fn = fn

    @property
    def data(self):
        if not hasattr(self, '_data'):
            with open(self.fn, 'rb') as f:
                self._data = pickle.load(f)

        return self._data

    def get_labels(self, pars):

        params, redshifts = self.data['pinfo']

        if type(pars[0]) == str:
            pass
        else:
            _pars = [params[i] for i in pars]
            pars = _pars

        labels = []

        for par in pars:
            if par in _default_labels:
                labels.append(_default_labels[par])
            else:
                labels.append(par)

        return labels

    def plot_triangle(self, fig=1, axes=None, elements=None,
        complement=False, bins=20, burn=0, fig_kwargs={}, contours=True,
        fill=False, nu=[0.95, 0.68], take_log=False, is_log=False,
        skip=None, **kwargs):
        """

        """

        has_ax = axes is not None

        if not has_ax:
            fig = pl.figure(constrained_layout=True, num=fig, **fig_kwargs)
            fig.subplots_adjust(hspace=0.05, wspace=0.05)
        else:
            axes_by_row = axes

        if elements is None:
            elements = np.arange(self.data['flatchain'].shape[-1])

        labels = self.get_labels(elements)
        Np = len(elements)

        if type(bins) not in [list, tuple, np.ndarray]:
            bins = [bins] * Np
        if type(complement) not in [list, tuple, np.ndarray]:
            complement = [complement] * Np
        if type(is_log) not in [list, tuple, np.ndarray]:
            is_log = [is_log] * Np
        if type(take_log) not in [list, tuple, np.ndarray]:
            take_log = [take_log] * Np

        # Remember, for gridspec, rows are numbered frop top-down.
        if not has_ax:
            gs = fig.add_gridspec(Np, Np)
            axes_by_row = [[] for i in range(Np)]

        flatchain = self.data['flatchain']

        for i, row in enumerate(range(Np)):
            for j, col in enumerate(range(Np)):
                # Skip elements in upper triangle
                if j > i:
                    continue

                if skip is not None:
                    if i in skip:
                        continue
                    if j in skip:
                        continue

                # Create axis
                if not has_ax:
                    _ax = fig.add_subplot(gs[i,j])
                    axes_by_row[i].append(_ax)
                else:
                    _ax = axes_by_row[i][j]

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
                    kw = kwargs.copy()
                    if 'colors' in kw:
                        del kw['colors']
                    _ax.hist(p2, density=True, bins=bins[j], histtype='step', **kw)

                    if j > 0:
                        _ax.set_yticklabels([])
                        if j == Np - 1:
                            _ax.set_xlabel(labels[j])
                        else:
                            _ax.set_xticklabels([])
                    else:
                        _ax.set_ylabel(r'PDF')

                    ok = np.isfinite(p2)
                    _ax.set_xlim(p2[ok==1].min(), p2[ok==1].max())
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

                ok1 = np.isfinite(p1)
                ok2 = np.isfinite(p2)
                _ax.set_ylim(p1[ok1==1].min(), p1[ok1==1].max())
                _ax.set_xlim(p2[ok2==1].min(), p2[ok2==1].max())

        # Done
        return fig, axes_by_row


    def plot_walker_trajectories(self, burn=0, ax=None, fig=1, **kwargs):

        params, redshifts = self.data['pinfo']

        nrows = len(_all_pars)

        ncols = max(self.data['zfit'].size,
            sum([par.startswith('Q') for par in params]))

        if ax is None:
            fig, axes = pl.subplots(nrows, ncols, num=fig,
                figsize=(ncols * 4, nrows *4))

            if ncols == 1:
                axes = [[ax] for ax in axes]

        steps = np.arange(0, self.data['chain'].shape[1])

        ibest = np.argwhere(self.data['lnprob'] == self.data['lnprob'].max())[-1]

        # This looks slow/lazy but it's to keep ordering.
        punique = []
        for i, par in enumerate(params):
            if par in punique:
                continue

            punique.append(par)

        zunique = np.unique(redshifts)
        zunique = zunique[np.isfinite(zunique)]

        ct = 0
        for i, par in enumerate(params):

            _z_ = redshifts[i]

            # Special cases: parametric elements of model
            if np.isinf(_z_):
                parname, parnum = par.split('_')

                _i = _all_pars.index(parname)

                _j = int(par[-1])
                ylab = par[0]

                axes[_i][_j].annotate(par, (0.05, 0.95), bbox=bbox,
                    xycoords='axes fraction', ha='left', va='top')

            else:
                _i = _all_pars.index(par)
                _j = np.argmin(np.abs(zunique - _z_))
                ylab = par

                axes[_i][_j].annotate(r'$z=%.2f$' % _z_, (0.05, 0.95),
                    xycoords='axes fraction', ha='left', va='top',
                    bbox=bbox)

            chain = self.data['chain'][:,burn:,i]

            axes[_i][_j].plot(steps, chain.T, **kwargs)

            # Plot best one as horizontal line
            axes[_i][_j].plot(steps,
                chain[ibest[0],ibest[1]] * np.ones_like(steps), color='k',
                ls='--', zorder=10, lw=3)

            # Put marker at walker/step where best happens
            axes[_i][_j].scatter(steps[ibest[1]], chain[ibest[0],ibest[1]],
                marker='|', s=150, color='k', zorder=10)

            if _j == 0:
                axes[_i][_j].set_ylabel(ylab)

        return axes

    def plot_ps(self, z=None, use_best=True, ax=None, fig=1, conflevel=0.68,
        marker_kw={}, use_cbar=True, show_cbar=True, show_data=True, cmap='jet',
        burn=0, **kwargs):
        """
        Plot the power spectrum, either at the maximum likelihood point or
        as a shaded region indicative of a given confidence level.
        """
        if ax is None:
            fig, ax = pl.subplots(1, 1, num=fig)

        if use_cbar:
            norm = Normalize(vmin=min(self.data['zfit']),
                vmax=max(self.data['zfit']))
            cmap = ScalarMappable(norm=norm, cmap=cmap)
            cmap.set_array([])

        burn_per_w = burn // self.data['chain'].shape[0]

        data = self.data
        ibest = np.argwhere(data['lnprob'] == data['lnprob'].max())[0]
        sh = data['blobs'].shape

        _ps = np.reshape(data['blobs'], (sh[0]*sh[1],sh[2],sh[3]))
        colors = 'k', 'b', 'm', 'c', 'y'
        for i in range(sh[2]):
            ps = _ps[:,i,:]
            _z_ = self.data['zfit'][i]

            if z is not None:
                if z != _z_:
                    continue

            if use_cbar:
                kwargs['color'] = cmap.to_rgba(_z_)

            if use_best:
                ax.plot(data['kblobs'], data['blobs'][ibest[1], ibest[0],i],
                    **kwargs)
            else:
                _lo = (1. - conflevel) * 100 / 2.
                _hi = 100 - _lo
                lo, hi = np.percentile(ps[burn:], (_lo, _hi), axis=0)
                ax.fill_between(data['kblobs'], lo, hi,
                    **kwargs)

        ##
        # Overplot data
        if ('data' in data.keys()) and show_data:

            # Use cmap to force match in colors
            for i, _z_ in enumerate(data['zfit']):
                if z is not None:
                    if z != _z_:
                        continue

                ydat, yerr = data['data'][i]

                if use_cbar:
                    marker_kw['color'] = cmap.to_rgba(_z_)

                ax.errorbar(data['kblobs'], ydat, yerr.T, **marker_kw)

        ax.set_xlabel(labels['k'])
        ax.set_ylabel(labels['delta_sq'])
        ax.set_xscale('log')
        ax.set_yscale('log')

        try:
            ax.set_ylim(data['blobs'].min()*0.5, data['blobs'].max() * 2)
        except:
            ax.set_ylim(1, 1e4)

        if use_cbar and show_cbar:
            cax = fig.add_axes([0.91, 0.11, 0.015, 0.77])
            cb = pl.colorbar(cmap, ax=ax, cax=cax, orientation='vertical')
            cb.set_label(r'$z$')


        return ax

    def get_par_from_increments(self):
        pass

    def plot_igm_constraints(self, z=None, use_best=False, conflevel=0.68,
        ax=None, fig=1, burn=0, marker_kw={}, scatter=False, zoffset=0,
        bins=20, smooth_hist=None, **kwargs):
        """
        Plot contours in Q-T_S space. Kind of the whole point of this package.
        """

        new_ax = False
        if ax is None:
            fig, ax = pl.subplots(1, 1, num=fig)
            new_ax = True



        params, redshifts = self.data['pinfo']

        for _z_ in self.data['zfit']:

            iT = None
            iQ = None
            for j, par in enumerate(params):


                if (z is not None) and (_z_ != z):
                    continue

                if (par == 'Ts'):
                    iT = j
                if (par == 'Q'):
                    iQ = j

            T = self.data['flatchain'][burn:,iT]
            Q = self.data['flatchain'][burn:,iQ]
            x = 1. - Q

            hist, be1, be2 = np.histogram2d(T, x, bins)
            bc1 = bin_e2c(be1)
            bc2 = bin_e2c(be2)

            if smooth_hist is not None:
                hist = gaussian_filter(hist, smooth_hist)

            nu, levels = get_error_2d(T, x, hist, [bc1, bc2], nu=conflevel)
            ax.contour(bc2, bc1, hist / hist.max(), levels, **kwargs)

        if new_ax:
            ax.set_xlabel(r'$x_{\mathrm{HI}} \equiv 1 - Q$')
            ax.set_ylabel(r'$T_S \ [\mathrm{K}]$')

        return ax

    def plot_zevol(self, par, use_best=False, conflevel=0.68, ax=None, fig=1,
        burn=0, marker_kw={}, scatter=False, zoffset=0, **kwargs):
        """
        Plot constraints on model parameters vs. redshift.
        """
        if ax is None:
            fig, ax = pl.subplots(1, 1, num=fig)

        params, redshifts = self.data['pinfo']

        chain = self.data['chain']
        burn_per_w = burn // self.data['chain'].shape[0]

        # May be repeats -- just take first one
        ibest = np.argwhere(self.data['lnprob'] == self.data['lnprob'].max())
        if ibest.ndim == 2:
            ibest = ibest[0]

        # First, deal with parametric results if we have them.
        for _par_ in ['Q', 'R']:
            if (par != _par_):
                continue

            if (self.data['kwargs']['{}_func'.format(_par_[0])] is None):
                continue

            j = 0
            p = []
            v = []
            while '{}_p{}'.format(_par_[0], j) in params:
                pj = params.index('{}_p{}'.format(_par_[0], j))
                vj = chain[:,:,pj]
                j += 1

                p.append(pj)
                v.append(vj)

            fname = self.data['kwargs']['{}_func'.format(_par_[0])]

            if fname == 'tanh':
                func = tanh_generic
            elif fname == 'pl':
                if _par_ == 'Q':
                    func = power_law_max1
                else:
                    func = power_law
            elif fname == 'bpl':
                if _par_ == 'Q':
                    func = broken_power_law_max1
                else:
                    func = broken_power_law
            else:
                raise NotImplemented('No option for {} yet'.format(fname))

            pbest = [element[ibest[0],ibest[1]] for element in v]


            # Make Q(z) for each MCMC sample
            if use_best:
                ybest = func(_default_z, pbest)
                ax.plot(_default_z, ybest, **kwargs)
            else:
                v_flat = [self.data['flatchain'][burn:,_p] for _p in p]
                _pars_ = np.array([element for element in v_flat])

                if scatter:
                    zplot = self.data['zfit']
                else:
                    zplot = _default_z

                ybest = func(zplot, pbest)

                y = np.zeros((v[0].size-burn, _default_z.size))
                for i, _z_ in enumerate(zplot):
                    y[:,i] = func(_z_, _pars_)

                _lo = (1. - conflevel) * 100 / 2.
                _hi = 100 - _lo
                lo, hi = np.percentile(y, (_lo, _hi), axis=0)

                if scatter:
                    kw = kwargs.copy()
                    for i, _z_ in enumerate(zplot):
                        ax.plot([_z_+zoffset]*2, [lo[i], hi[i]], **kw)
                        ax.scatter([_z_+zoffset]*2, [ybest[i]]*2, **marker_kw)

                        if 'label' in kw:
                            del kw['label']
                else:
                    ax.fill_between(zplot, lo, hi, **kwargs)

            ax.set_xlabel(r'$z$')
            ax.set_ylabel(_default_labels[_par_])
            ax.set_xlim(5, 20)
            ax.set_ylim(*_default_limits[_par_])

            return ax

        ##
        # Non-parametric results
        for i, _par_ in enumerate(params):

            z = redshifts[i]

            if (_par_ != par):
                continue

            #if (_par_ == 'R') and (self.data['kwargs']['Rxdelta'] is not None):
            #    continue
            #else:
            best = chain[ibest[0], ibest[1],i]
            _chain = self.data['flatchain'][burn:,i]

            # Get band
            _lo = (1. - conflevel) * 100 / 2.
            _hi = 100 - _lo
            lo, hi = np.percentile(_chain, (_lo, _hi), axis=0)
            ax.plot([z+zoffset]*2, [lo, hi], **kwargs)
            ax.scatter([z+zoffset]*2, [best]*2, **marker_kw)

        ax.set_xlabel(r'$z$')
        ax.set_ylabel(_default_labels[par])
        ax.set_xlim(5, 20)
        ax.set_ylim(*_default_limits[par])

        return ax

    def plot_loglike(self, burn=0, ax=None, fig=1, **kwargs):
        if ax is None:
            fig, ax = pl.subplots(1, 1, num=fig)

        burn_per_w = burn // self.data['blobs'].shape[1]
        x = np.arange(self.data['chain'].shape[1])
        for i in range(self.data['blobs'].shape[1]):
            ax.plot(x, self.data['lnprob'][i,burn_per_w:], **kwargs)

        ax.set_xlabel('step number')
        ax.set_ylabel(r'$\log \mathcal{L}$')

        return ax
