"""

analysis.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import pickle
import numpy as np
from .models import BubbleModel
from scipy.ndimage import gaussian_filter
from .inference import tanh_generic, power_law, power_law_max1, \
    broken_power_law, broken_power_law_max1, double_power_law, \
    extract_params, power_law_lognorm, erf_Q, power_law_Q, lin_Q
from .util import labels, bin_e2c, bin_c2e, get_error_2d

try:
    import matplotlib.pyplot as pl
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, Normalize
except ImportError:
    pass

_default_modes = np.logspace(-1, 0., 21)
_default_colors = ['k', 'b', 'm', 'c', 'r', 'g', 'y', 'orange']
_default_ls = ['-', '--', '-.', ':']
_default_labels = {'Q': r'$Q$', 'R': r'$R$', 'Ts': r'$T_S$',
    'sigma': r'$\sigma$', 'gamma': r'$\gamma$', 'Asys': r'$A_{\rm{sys}}$'}
_default_limits = {'Q': (-0.05, 1.05), 'R': (3e-1, 30),
    'Ts': (1, 200),
    'sigma': (0, 2), 'gamma': (-4, -2), 'Asys': (0.2, 2)}
_default_z = np.arange(5, 20, 0.05)
_default_Q = np.arange(0, 1.01, 0.01)

bbox = dict(facecolor='none', edgecolor='k', fc='w',
        boxstyle='round,pad=0.3', alpha=0.9, zorder=1000)

class AnalyzeFit(object): # pragma: no cover
    def __init__(self, fn):
        self.fn = fn

    @property
    def data(self):
        if not hasattr(self, '_data'):
            with open(self.fn, 'rb') as f:
                self._data = pickle.load(f)

        return self._data

    @property
    def model(self):
        if not hasattr(self, '_model'):
            self._model = BubbleModel(**self.data['kwargs'])
        return self._model

    @property
    def custom_labels(self):
        if not hasattr(self, '_custom_labels'):
            self._custom_labels = {}

        return self._custom_labels

    @custom_labels.setter
    def custom_labels(self, value):
        if not hasattr(self, '_custom_labels'):
            self._custom_labels = {}
        self._custom_labels.update(value)

    def get_labels(self, pars, redshifts=None):

        labels = []

        for i, par in enumerate(pars):
            if par in self.custom_labels:
                labels.append(self.custom_labels[par])
            elif par in _default_labels:
                if redshifts is not None:
                    s = _default_labels[par]
                    j = s.find('$')
                    k = s.rfind('$')
                    l = s[j+1:k]
                    lab = r'$%s(z=%.2f)$' % (l, redshifts[i])
                else:
                    lab = _default_labels[par]
                labels.append(lab)
            else:
                labels.append(par)

        return labels

    def check_model_ps(self, z=None, k=None, Ts=(0, np.inf), Q=(0, 1), R=(0, 100),
        sigma=(0.5, 2), gamma=(-4, 0), skip=0):
        """
        Scroll through models in some specified corner of parameter space
        and re-compute the power spectrum and plot it.
        """

        pars, redshifts = self.data['pinfo']
        fchain = self.data['flatchain']

        if np.unique(redshifts).size > 1:
            assert z is not None, "Must supply `z` if multi-z fit."

        limits = {'Ts': Ts, 'Q': Q, 'R': R, 'sigma': sigma, 'gamma': gamma}

        if k is None:
            k = np.logspace(-1, 0, 11)

        for i in range(fchain.shape[0]):
            if i < skip:
                continue

            kw = {par:fchain[i,j] for j, par in enumerate(pars)}
            if self.data['kwargs']['Ts_log10']:
                kw['Ts'] = 10**kw['Ts']

            plot_it = True
            for par in pars:
                if limits[par][0] < kw[par] < limits[par][1]:
                    continue

                plot_it = False

            if not plot_it:
                continue

            print('Generating model from chain link {} with kwargs={}'.format(i,
                kw))
            ps = self.model.get_ps_21cm(redshifts[0], k, **kw)

            pl.loglog(k, k**3 * ps / 2. / np.pi**2)

            input('<enter> for next model')
            pl.clear()

    def plot_triangle(self, fig=1, axes=None, params=None, redshifts=None,
        complement=False, bins=20, burn=0, fig_kwargs={}, contours=True,
        fill=False, nu=[0.95, 0.68], take_log=False, is_log=False,
        skip=None, smooth=None, skip_params=None, show_1d=True,
        logL_cut=-np.inf, **kwargs):
        """

        """

        has_ax = axes is not None

        if not has_ax:
            fig = pl.figure(constrained_layout=True, num=fig, **fig_kwargs)
            fig.subplots_adjust(hspace=0.05, wspace=0.05)
        else:
            axes_by_row = axes

        all_params, redshifts = self.data['pinfo']

        if params is None:
            params = all_params
        else:
            pass

        elements = range(len(params))

        try:
            labels = self.get_labels(params, redshifts)
        except IndexError:
            labels = [''] * len(params)
        Np = len(params)

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
            if show_1d:
                gs = fig.add_gridspec(Np, Np)
                axes_by_row = [[] for i in range(Np)]
            else:
                axes_by_row = [fig.add_subplot(111)]

        flatchain = self.data['flatchain']

        for i, row in enumerate(range(Np)):
            for j, col in enumerate(range(Np)):
                # Skip elements in upper triangle
                if j > i:
                    continue

                if (not show_1d) and i == j:
                    continue

                if skip is not None:
                    if i in skip:
                        continue
                    if j in skip:
                        continue

                # Create axis
                if show_1d:
                    if not has_ax:
                        _ax = fig.add_subplot(gs[i,j])
                        axes_by_row[i].append(_ax)
                    else:
                        _ax = axes_by_row[i][j]
                else:
                    _ax = axes_by_row[0]

                if skip_params is not None:
                    if params[i] in skip_params:
                        continue

                    if params[j] in skip_params:
                        continue

                if params[i] not in all_params:
                    continue
                if params[j] not in all_params:
                    continue

                zsamplesi, samplesi = self.get_samples(params[i], burn)
                zsamplesj, samplesj = self.get_samples(params[j], burn)

                if zsamplesi.size > 1:
                    iz = np.argmin(np.abs(redshifts[i] - zsamplesi))
                    idata = samplesi[:,iz]
                else:
                    idata = samplesi[0,:]

                if zsamplesj.size > 1:
                    jz = np.argmin(np.abs(redshifts[j] - zsamplesj))
                    jdata = samplesj[:,jz]
                else:
                    jdata = samplesj[0,:]

                # Retrieve data to be used in plot
                if not is_log[i]:
                    p1 = 1. - idata if complement[i] else idata
                else:
                    p1 = 10**idata if is_log[i] else idata

                if take_log[i]:
                    p1 = np.log10(p1)

                # 2-D PDFs from here on
                if not is_log[j]:
                    p2 = 1. - jdata if complement[j] else jdata
                else:
                    p2 = 10**jdata if is_log[j] else jdata

                if take_log[j]:
                    p2 = np.log10(p2)

                # 1-D PDFs
                if i == j:

                    kw = kwargs.copy()
                    if 'colors' in kw:
                        del kw['colors']
                    if 'linestyles' in kw:
                        del kw['linestyles']

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

                    if smooth is not None:
                        hist = gaussian_filter(hist, smooth)

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

        nrows = len(params)
        ncols = 1
        _j = 0

        if ax is None:
            fig, axes = pl.subplots(nrows, ncols, num=fig,
                figsize=(ncols * 4, nrows * 4))

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

            try:
                parname, parnum = par.split('_')
            except ValueError:
                parname = par
                parnum = None

            axes[i].annotate(par, (0.05, 0.95), bbox=bbox,
                xycoords='axes fraction', ha='left', va='top')

            chain = self.data['chain'][:,burn:,i]

            axes[i].plot(steps, chain.T, **kwargs)

            # Plot best one as horizontal line
            axes[i].plot(steps,
                chain[ibest[0],ibest[1]] * np.ones_like(steps), color='k',
                ls='--', zorder=10, lw=3)

            # Put marker at walker/step where best happens
            axes[i].scatter(steps[ibest[1]], chain[ibest[0],ibest[1]],
                marker='|', s=150, color='k', zorder=10)

            if _j == 0:
                ylab = par.split('_')[0]
                axes[i].set_ylabel(ylab)

        return axes

    def get_zindex_in_data(self, z, ztol=1e-3):
        j = np.argmin(np.abs(z - self.data['zfit']))
        assert abs(self.data['zfit'][j] - z) < ztol

        return j

    def get_ps(self, z=None, ztol=1e-2, burn=0, reshape=True):
        burn_per_w = burn // self.data['chain'].shape[0]

        data = self.data
        sh = data['blobs'].shape
        try:
            nsteps, nw, nz, nk = sh
        except ValueError:
            nsteps, nw, nz = sh
            nk = 1

        if reshape:
            _ps = np.reshape(data['blobs'], (nsteps*nw,nz,nk))
        else:
            _ps = data['blobs']

        if z is not None:
            i = self.get_zindex_in_data(z, ztol=ztol)

            if reshape:
                ps = _ps[:,i]
            else:
                ps = _ps[:,:,i]

            return self.data['kblobs'], ps[burn_per_w:]
        else:
            return self.data['kblobs'], _ps[burn_per_w:]

    def get_stuck_walkers(self, logL_min=-np.inf, burn=0):
        nw = self.data['chain'].shape[0]
        burn_per_w = burn // nw

        bad_L = self.data['lnprob'][:,burn_per_w:] < logL_min
        bad_w = np.any(bad_L, axis=1)

        bad_i = np.argwhere(bad_w).squeeze()

        return bad_i

    def plot_ps(self, z=None, use_best=True, ax=None, fig=1, ztol=1e-2,
        conflevel=0.68, samples=None, show_recovery=True,
        marker_kw={}, use_cbar=True, show_cbar=True, show_data=True, cmap='jet',
        burn=0, logL_min=-np.inf, logL_max=np.inf, **kwargs):
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

        sh = self.data['blobs'].shape
        try:
            nsteps, nw, nz, nk = sh
        except ValueError:
            nsteps, nw, nz = sh
            nk = 1

        burn_per_w = burn // nw
        ibest = np.argwhere(self.data['lnprob'] == self.data['lnprob'].max())[0]

        i = self.get_zindex_in_data(z=z, ztol=ztol)
        _z_ = self.data['zfit'][i]

        if (logL_min > -np.inf):
            bad_i = self.get_stuck_walkers(logL_min, burn=burn)
            if bad_i.size == nw:
                raise ValueError("No acceptable walkers!")
            elif bad_i.size > 0:
                _k, _ps_ = self.get_ps(z=_z_, ztol=ztol, burn=burn, reshape=False)

                _ps = []
                for iw in np.arange(nw):
                    if iw in bad_i:
                        continue

                    _ps.append(_ps_[:,iw,:])

                _ps = np.array(_ps)
                ps = np.reshape(_ps, (_ps.shape[0] * _ps.shape[1], nk))
            else:
                _k, ps = self.get_ps(z=_z_, ztol=ztol, burn=burn, reshape=True)
        else:
            _k, ps = self.get_ps(z=_z_, ztol=ztol, burn=burn, reshape=True)

        if use_cbar:
            kwargs['color'] = cmap.to_rgba(_z_)

        if not show_recovery:
            pass
        elif use_best:
            ax.plot(self.data['kblobs'], self.data['blobs'][ibest[1],
                ibest[0],i], **kwargs)
        elif samples is not None:
            ax.plot(self.data['kblobs'], ps[-samples:,:].T, **kwargs)
        else:
            _lo = (1. - conflevel) * 100 / 2.
            _hi = 100 - _lo
            lo, hi = np.nanpercentile(ps, (_lo, _hi), axis=0)
            ax.fill_between(self.data['kblobs'], lo, hi, **kwargs)

        ##
        # Overplot data
        if ('data' in self.data.keys()) and show_data:

            # Use cmap to force match in colors
            ydat, yerr = self.data['data'][i]

            if use_cbar:
                marker_kw['color'] = cmap.to_rgba(_z_)

            ax.errorbar(self.data['kblobs'], ydat, yerr.T, **marker_kw)

        ax.set_xlabel(labels['k'])
        ax.set_ylabel(labels['delta_sq'])
        ax.set_xscale('log')
        ax.set_yscale('log')

        try:
            ax.set_ylim(self.data['blobs'].min()*0.5, self.data['blobs'].max() * 2)
        except:
            ax.set_ylim(1, 1e4)

        if use_cbar and show_cbar and show_recovery:
            cax = fig.add_axes([0.91, 0.11, 0.015, 0.77])
            cb = pl.colorbar(cmap, ax=ax, cax=cax, orientation='vertical')
            cb.set_label(r'$z$', fontsize=20)


        return fig, ax

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

    def get_samples(self, par, burn=0):

        params, redshifts = self.data['pinfo']

        chain = self.data['chain']
        burn_per_w = burn // self.data['chain'].shape[0]

        ##
        # If par in `params`, it's easy.
        if par in params:
            z = []
            y = []
            for i, _z_ in enumerate(redshifts):
                if params[i] != par:
                    continue

                z.append(_z_)
                y.append(self.data['flatchain'][burn:,i])

            return np.array(z), np.array(y)

        # May be repeats -- just take first one
        ibest = np.argwhere(self.data['lnprob'] == self.data['lnprob'].max())
        if ibest.ndim == 2:
            ibest = ibest[0]

        # First, deal with parametric results if we have them.
        for _par_ in ['Q', 'R', 'Ts']:
            if (par != _par_):
                continue

            if (self.data['kwargs']['{}_func'.format(_par_)] is None):
                continue

            j = 0
            p = []
            v = []
            while '{}_p{}'.format(_par_, j) in params:
                pj = params.index('{}_p{}'.format(_par_, j))
                vj = chain[:,:,pj]
                j += 1

                p.append(pj)
                v.append(vj)

            fname = self.data['kwargs']['{}_func'.format(_par_)]

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
            elif fname == 'dpl':
                assert _par_ == 'Ts'
                func = double_power_law
            else:
                raise NotImplemented('No option for {} yet'.format(fname))

            # Make Q(z) (for example) for each MCMC sample
            v_flat = [self.data['flatchain'][burn:,_p] for _p in p]
            _pars_ = np.array([element for element in v_flat])
            zplot = self.data['zfit']

            assert burn < v[0].size, \
                "Provided `burn` exceeds size of chain!"

            y = np.zeros((v[0].size-burn, zplot.size))
            for i, _z_ in enumerate(zplot):
                y[:,i] = func(_z_, _pars_)

            return np.array(zplot), y

    def plot_evol(self, par, use_best=False, conflevel=0.68,
        ax=None, fig=1, burn=0, marker_kw={}, scatter=False, boxplot=False,
        zoffset=0, samples=None, **kwargs):
        """
        Plot constraints on model parameters vs. redshift.
        """

        if ax is None:
            fig, ax = pl.subplots(1, 1, num=fig)

        params, redshifts = self.data['pinfo']

        chain = self.data['chain']

        # May be repeats -- just take first one
        ibest = np.argwhere(self.data['lnprob'] == self.data['lnprob'].max())
        if ibest.ndim == 2:
            ibest = ibest[0]

        # First, deal with parametric results if we have them.
        for _par_ in ['Q', 'R', 'Ts', 'sigma', 'Asys']:
            if (par != _par_):
                continue

            if (self.data['kwargs']['{}_func'.format(_par_)] is None):
                continue

            j = 0
            p = []
            v = []
            while '{}_p{}'.format(_par_, j) in params:
                pj = params.index('{}_p{}'.format(_par_, j))
                vj = chain[:,:,pj]
                j += 1

                p.append(pj)
                v.append(vj)

            fname = self.data['kwargs']['{}_func'.format(_par_)]

            if fname == 'tanh':
                func = tanh_generic
            elif fname == 'linear':
                func = lin_Q
            elif fname == 'pl':
                if _par_ == 'Q':
                    func = power_law_max1
                elif _par_ == 'Ts':
                    func = power_law_lognorm
                elif _par_ == 'R':
                    func = power_law_Q
                else:
                    func = power_law
            elif fname == 'bpl':
                if _par_ == 'Q':
                    func = broken_power_law_max1
                else:
                    func = broken_power_law
            elif fname == 'dpl':
                assert _par_ == 'Ts'
                func = double_power_law
            elif fname == 'erf':
                assert _par_ == 'Asys'
                func = erf_Q
            else:
                raise NotImplemented('No option for {} yet'.format(fname))

            pbest = [element[ibest[0],ibest[1]] for element in v]

            # Make Q(z) for each MCMC sample
            if use_best:
                if par in ['R', 'sigma', 'Asys']:
                    ybest = func(_default_Q, pbest)
                    ax.plot(_default_Q, ybest, **kwargs)
                else:
                    ybest = func(_default_z, pbest)
                    ax.plot(_default_z, ybest, **kwargs)
            else:
                v_flat = [self.data['flatchain'][burn:,_p] \
                    for _p in p]
                _pars_ = np.array([element for element in v_flat])

                if par in ['R', 'Asys']:
                    ybest = func(_default_Q, pbest)
                    xvals = _default_Q
                else:
                    if scatter or boxplot:
                        zplot = self.data['zfit']
                    else:
                        zplot = _default_z

                    ybest = func(zplot, pbest)
                    xvals = zplot

                assert burn < v[0].size, \
                    "Provided `burn` exceeds size of chain!"

                y = np.zeros((v[0].size-burn, xvals.size))
                for i, _x_ in enumerate(xvals):
                    y[:,i] = func(_x_, _pars_)

                #zplot, y = self.get_samples

                _lo = (1. - conflevel) * 100 / 2.
                _hi = 100 - _lo
                lo, hi = np.percentile(y, (_lo, _hi), axis=0)

                if boxplot:
                    kw = kwargs.copy()
                    for i, _x_ in enumerate(xvals):
                        conf = np.array([[_lo / 100., _hi / 100.]])
                        data = y[:,i]#np.concatenate((y[:,i], ybest[i]))
                        ax.boxplot(data, positions=[_x_],
                            showfliers=False,
                            manage_ticks=False,
                            conf_intervals=conf)
                            #usermedians=[ybest[i]])
                            #conf_intervals=[[16, 84]])

                        if 'label' in kw:
                            del kw['label']
                elif scatter:
                    kw = kwargs.copy()
                    for i, _x_ in enumerate(zplot):
                        ax.plot([_x_+zoffset]*2, [lo[i], hi[i]], **kw)
                        ax.scatter([_x_+zoffset]*2, [ybest[i]]*2, **marker_kw)

                        if 'label' in kw:
                            del kw['label']

                elif samples is not None:
                    ax.plot(xvals, y[-samples:,:].T, **kwargs)
                else:
                    ax.fill_between(xvals, lo, hi, **kwargs)

            if par in ['Ts', 'R']:
                ax.set_yscale('log')

            if par in ['R', 'sigma', 'Asys']:
                ax.set_xlabel(r'$Q$')
                ax.set_xlim(0, 1)
            else:
                ax.set_xlabel(r'$z$')
                ax.set_xlim(min(self.data['zfit'])-1,
                    max(self.data['zfit'])+1)

            ax.set_ylabel(_default_labels[_par_])

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

            if _par_ == 'Ts' and self.data['kwargs']['Ts_log10']:
                lo = 10**lo
                hi = 10**hi
                best = 10**best

            ax.plot([z+zoffset]*2, [lo, hi], **kwargs)
            ax.scatter([z+zoffset]*2, [best]*2, **marker_kw)

        if par in ['Ts', 'R']:
            ax.set_yscale('log')

        ax.set_xlabel(r'$z$')
        ax.set_ylabel(_default_labels[par])
        ax.set_xlim(min(self.data['zfit'])-1,
            max(self.data['zfit'])+1)
        ax.set_ylim(*_default_limits[par])

        return ax

    def plot_loglike(self, burn=0, ax=None, fig=1, **kwargs):
        if ax is None:
            fig, ax = pl.subplots(1, 1, num=fig)

        burn_per_w = burn // self.data['chain'].shape[1]
        x = np.arange(self.data['chain'].shape[1])
        for i in range(self.data['chain'].shape[0]):
            ax.plot(x, self.data['lnprob'][i,burn_per_w:], **kwargs)

        ax.axhline(self.data['lnprob'][np.isfinite(self.data['lnprob'])].max(),
            color='k', ls='--', lw=3, zorder=10)

        ax.set_xlabel('step number')
        ax.set_ylabel(r'$\log \mathcal{L}$')

        return ax
