"""

box.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed  4 Aug 2021 11:41:40 EDT

Description: Make 3-d realizations of our bubble model.

"""

import os
import numpy as np
import matplotlib.pyplot as pl
from .models import BubbleModel
from scipy.spatial import cKDTree
from .util import smooth_box, ProgressBar, bin_c2e

try:
    import powerbox as pbox
except ImportError:
    pass

try:
    import h5py
except ImportError:
    pass

class Box(BubbleModel):
    """
    A class for making 3-d realizations of our phenomenological model or other
    toy models for reionization.
    """
    def __init__(self, **kwargs):
        """
        Treat this just like a BubbleModel instance.
        """
        BubbleModel.__init__(self, **kwargs)

    def get_box_path(self, Q, z=None, which_box='bubble',
        allow_partial_ionization=0, path='.', Lbox=100., vox=1., **kwargs):

        path = '{}/boxes_R_{:.1f}'.format(path, kwargs['R'])

        if self.bubbles_pdf == 'lognormal':
            path += '_sigma_{:.2f}'.format(kwargs['sigma'])
        else:
            path += '_gamma_{:.2f}'.format(kwargs['gamma'])

        if which_box != 'bubbles':
            path += '_z_{:.2f}'.format(z)

        if which_box == '21cm':
            path += '_Ts_{:.1f}'.format(kwargs['Ts'])

        return path

    def get_box_name(self, Q, which_box='bubble', Lbox=100, vox=1., seed=None):
        fn = 'box_{}_L{:.0f}_v_{:.1f}_Q_{:.2f}_seed_{}'.format(which_box,
            Lbox, vox, Q, seed)

        return fn

    def generate_boxes(self, Q, z=None, which_box='bubble',
        allow_partial_ionization=0, path='.',
        seeds=None, Lbox=100., vox=1., clobber=False, **kwargs):
        """
        Generate a series of boxes at different Q's so we can save time later
        and just load them from disk.

        Parameters
        ----------
        Q : np.ndarray
            Array of Q values at which to run boxes.
        which_box : str
            Can be 'bb', 'density', or '21cm' at this point.
        allow_partial_ionization : bool
            Whether or not to attempt to include subgrid scheme for ionization.

        All additional kwargs are those getting passed to the `get_ps_<whatever>`
        method of our inherited modeling instance.

        """

        if which_box == 'bubbles':
            box_func = self.get_box_bubbles
        elif which_box == 'density':
            box_func = self.get_box_density
            assert z is not None, "Must pass `z` for density box!"
        elif which_box == '21cm':
            box_func = self.get_box_21cm
            assert z is not None, "Must pass `z` for 21-cm box!"
        elif which_box == 'noise':
            box_func = self.get_box_rand
        else:
            raise NotImplemented('help')

        if seeds is not None:
            assert len(seeds) == len(Q)
        else:
            seeds = [None] * len(Q)

        path = self.get_box_path(Q, z=z, which_box=which_box,
            allow_partial_ionization=allow_partial_ionization, path=path,
            seeds=seeds, Lbox=Lbox, vox=vox, **kwargs)

        if not os.path.exists(path):
            os.mkdir(path)

        pb = ProgressBar(Q.size, name='box(Q)')
        pb.start()

        # Loop over Q and save boxes
        for i, _Q_ in enumerate(Q):
            pb.update(i)

            fn = path + '/' \
                + self.get_box_name(_Q_, which_box=which_box, Lbox=Lbox, vox=vox) \
                + '.hdf5'

            if os.path.exists(fn) and (not clobber):
                print("Found box {}. Moving on...".format(fn))
                continue

            box = box_func(z, Q=_Q_, Lbox=Lbox, vox=vox,
                allow_partial_ionization=allow_partial_ionization, seed=seeds[i],
                **kwargs)

            with h5py.File(fn, 'w') as f:
                f.create_dataset(which_box, data=box)

            print("Wrote {}.".format(fn))

        pb.finish()

    def load_box(self, path='.', Lbox=100., vox=1., Q=0.0, Ts=np.inf,
        R=5., sigma=0.5, gamma=0., use_kdtree=True, which_box='bubble',
        allow_partial_ionization=True, z=None, seed=None):

        path = self.get_box_path(Q, z=z, which_box=which_box,
            allow_partial_ionization=allow_partial_ionization, path=path,
            seed=seed, Lbox=Lbox, vox=vox, Ts=Ts, R=R, sigma=sigma, gamma=gamma)

        fn = path + '/' \
            + self.get_box_name(Q, which_box=which_box, Lbox=Lbox, vox=vox,
                seed=seed) \
            + '.hdf5'

        if os.path.exists(fn):
            with h5py.File(fn, 'r') as f:
                data = np.array(f[(which_box)])

            print("Read box from {}.".format(fn))

            return data
        else:
            print("No pre-existing box in {}.".format(fn))

        return None

    def get_box_density(self, z, vox=1., Lbox=100.):
        """
        Create a density box using Steven Murray's `powerbox` package.
        """

        box_disk = self.load_box(z=z, Lbox=Lbox, vox=vox, which_box='density')

        if box_disk is not None:
            return box_disk

        power = lambda k: self.get_ps_matter(z=z, k=k)

        Npix = int(Lbox / vox)
        assert Lbox / vox % 1 == 0
        rho = pbox.LogNormalPowerBox(N=Npix, dim=3, pk=power,
            boxlength=Lbox).delta_x()

        return rho

    def get_box_21cm(self, z, Lbox=100., vox=1., Q=0.0, Ts=np.inf,
        R=5., sigma=0.5, gamma=0., use_kdtree=True, path='.',
        allow_partial_ionization=True, seed=None):

        box_disk = self.load_box(path=path, Q=Q, z=z, which_box='21cm',
            allow_partial_ionization=allow_partial_ionization,
            seed=seed, Lbox=Lbox, vox=vox)

        if box_disk is not None:
            return box_disk

        xHI, Nb = self.get_box_bubbles(z, Lbox=Lbox, vox=vox, Q=Q,
            R=R, sigma=sigma, gamma=gamma, use_kdtree=use_kdtree, seed=seed,
            allow_partial_ionization=allow_partial_ionization)

        # Set bulk IGM temperature
        T0 = self.get_dTb_bulk(z, Ts=Ts)

        # Density box, no correlation with bubble box.
        rho = self.get_box_density(z, vox, Lbox)

        # Brightness temperature box
        dTb = T0 * xHI * (1. + rho)

        return dTb

    def _cache_box(self, field, args):

        if not hasattr(self, '_cache_box_'):
            self._cache_box_ = {}

        if field not in self._cache_box_:
            self._cache_box_[field] = {}
        if args in self._cache_box_[field]:
            return self._cache_box_[field][args]
        else:
            return None

    def get_box_bubbles(self, z, Lbox=100., vox=1., Q=0.5,  R=5., sigma=0.5,
        gamma=0., use_kdtree=True, allow_partial_ionization=False, seed=None,
        path='.'):
        """
        Make a 3-d realization of the bubble field.

        .. note :: This just draws bubbles from the desired bubble size
            distribution and positions them randomly in a box.

        Parameters
        ----------
        z : int, float
            Redshift of interest. Only matters if include_rho=True.
        Lbox : int, float
            Linear dimension of box to 'simulate' in [cMpc / h].
        vox : int, float
            Linear dimension of voxels in [cMpc / h].
        use_kdtree : bool
            If True, uses kdtree to speed-up placement of bubbles in volume.
        include_rho : bool
            If True, use Steven Murray's powerbox package to generate a 3-D
            realization of the density field and multiply box by (1 + delta).

        Returns
        -------
        A tuple of two elements: first, just the ionization box (ones and zeros),
        and second, a box containing the number of bubbles that engulf every
        single point, which can have any integer value > 0. The latter is
        used largely for diagnosing overlap. Each is a 3-d array with dimensions
        [Lbox / vox]*3.

        """

        args = (z, Lbox, vox, Q, R, sigma, gamma, use_kdtree,
            allow_partial_ionization, seed)

        cached_result = self._cache_box('bubbles', args)

        if cached_result is not None:
            print("Loaded box from cache.", args)
            return cached_result

        box_disk = self.load_box(path=path, Q=Q, z=z, which_box='bubbles',
            allow_partial_ionization=allow_partial_ionization,
            seed=seed, Lbox=Lbox, vox=vox, R=R, sigma=sigma, gamma=gamma)

        if box_disk is not None:
            return box_disk

        assert vox == 1, "There's a bug for vox != 1 right now :("

        Npix = int(Lbox / vox)
        Vpix = vox**3

        pdf = self.get_bsd(Q=Q, R=R, sigma=sigma, gamma=gamma)
        cdf = self.get_bsd_cdf(Q=Q, R=R, sigma=sigma, gamma=gamma)
        num_per_vol = self.get_nb(Q=Q, R=R, sigma=sigma, gamma=gamma)

        num = int(num_per_vol * Lbox**3)

        bins = np.arange(0, Lbox+vox, vox)
        binc = np.arange(0.5*vox, Lbox, vox)

        xx, yy, zz = np.meshgrid(binc, binc, binc)

        # Randomly generate `num` bubbles with sizes drawn from BSD.
        np.random.seed(seed)
        n = np.random.rand(num)
        R_r = np.exp(np.interp(np.log(n), np.log(cdf), np.log(self.tab_R)))

        # Randomly generate (x, y, z) positions for all bubbles
        p_len = np.random.rand(num*3).reshape(num, 3) * Lbox
        # Get bubble positions in terms of array indices
        p_bin = np.digitize(p_len, bins) - 1

        # Initialize a box. We'll zero-out elements lying within bubbles below.
        box = np.ones([binc.size]*3)
        box_tot = np.zeros([binc.size]*3)

        # Can speed things up with a kdtree if you want.
        if use_kdtree:
            pos = np.array([xx.ravel(), yy.ravel(), zz.ravel()]).T
            kdtree = cKDTree(pos, boxsize=Lbox)

        # Loop over bubbles and flag all cells within them
        for h in range(p_bin.shape[0]):

            ##
            # Brute force: would not recommend.
            if not use_kdtree:
                i, j, k = p_bin[h]
                dr = np.sqrt((xx - xx[i,j,k])**2 + (yy - yy[i,j,k])**2 \
                   + (zz - zz[i,j,k])**2)
                in_bubble = dr <= R_r[h]
                box[in_bubble] = 0
                box_tot[in_bubble] += 1
                continue

            ##
            # Speed-up with kdtree
            p = p_bin[h]

            # `nearby` are indices in `pos`, i.e., not (i, j, k) indices
            d, nearby = kdtree.query(p, k=1e4, distance_upper_bound=R_r[h] * 1.5)

            in_bubble = d <= R_r[h]

            # Possible that only one point is in bubble
            if len(in_bubble == 1) == 1:
                if not allow_partial_ionization:
                    continue

            for elem in nearby[in_bubble==True]:
                a, b, c = pos[elem]
                i, j, k = np.digitize([a, b, c], bins) - 1

                if allow_partial_ionization:
                    Vb = 4. * np.pi * R_r[h]**3 / 3.

                    tmp = 1 * box[i,j,k]
                    tmp -= Vb / Vpix
                    box[i,j,k] = max(0, tmp)
                    box_tot[i,j,k] += Vb / Vpix
                else:
                    box[i,j,k] = 0
                    box_tot[i,j,k] += 1

        self._cache_box_['bubbles'][args] = box, box_tot

        return box, box_tot

    def get_box_rand(self, box=None, Lbox=100., vox=1., Q=0.5, Qtol=1e-2,
        seed=None):

        if box is None:
            Npix = int(Lbox / vox)
            box = np.zeros([Npix]*3)

        np.random.seed(seed)
        r = np.random.rand(box.size)
        box_r = np.array(r > Q, dtype=float).reshape(*box.shape)
        Qact = 1 - box_r.sum() / float(box.size)

        if abs(Qact - Q) > Qtol:
            raise ValueError("Requested Q={}, actual Q={}. Increase number of pixels?".format(Q, Qact))

        return box_r

    def plot_variance_vs_Q(self, Qarr=None, Rsm=1, R=5., sigma=1., gamma=0.,
        Lbox=100., vox=1., seed=None, fig=1, ax=None, show_random=True,
        show_analytic=True, allow_partial_ionization=False, **kwargs): # pragma: no cover

        has_ax = True
        if ax is None:
            fig, ax = pl.subplots(1, 1, num=fig)
            has_ax = False
        if Qarr is None:
            Qarr = np.arange(0.1, 1.1, 0.2)

        pb = ProgressBar(Qarr.size, name="ps(bb;Q)")
        pb.start()

        var = []
        var_r = []
        var_a = []
        for i, _Q_ in enumerate(Qarr):
            pb.update(i)

            box, box_tot = self.get_box_bubbles(z=np.inf, Lbox=Lbox, vox=vox,
                Q=_Q_, R=R, sigma=sigma, gamma=gamma, use_kdtree=True, seed=seed,
                allow_partial_ionization=allow_partial_ionization)

            box_sm = smooth_box(box, R=Rsm, periodic=True).real
            var.append(np.std(box_sm.ravel())**2)

            if show_random:
                box_r = self.get_box_rand(box, Q=_Q_)
                box_sm = smooth_box(box_r, R=Rsm, periodic=True).real
                var_r.append(np.std(box_sm.ravel())**2)

            if show_analytic:
                _var = self.get_variance_bb(z=np.inf, r=Rsm, Q=_Q_, R=R,
                    sigma=sigma, gamma=gamma)
                var_a.append(_var)

        pb.finish()

        ax.scatter(Qarr, var, **kwargs)

        if show_analytic:
            ax.plot(Qarr, var_a, **kwargs)

        if show_random and (not has_ax):
            norm = max(var) / max(var_r)
            ax.plot(Qarr, np.array(var_r) * norm, color='k', ls=':',
                label=r'noise (re-scaled)')

        ax.set_xlabel(r'$Q$')
        ax.set_ylabel(r'$\sigma_b^2$')
        ax.set_xlim(-0.05, 1.05)
        ax.legend()

        return ax


    def plot_variance_vs_scale(self, box, Rsm=None, rescale=1., fig=1, ax=None,
        show_random=True, **kwargs): # pragma: no cover
        """
        Plot the variance of a box vs. smoothing scale.
        """

        # If no array of smoothing scales provided, default to single pixel to
        # whole domain
        if Rsm is None:
            Rsm = np.arange(1, box.shape[0])

        if ax is None:
            fig, ax = pl.subplots(1, 1, num=fig)

        # Only makes sense if ionization box
        if show_random:
            Q = box[box == 0].size / float(box.size)
            box_r = self.get_box_rand(box=box, Q=Q)
            print('random box with Q={}...'.format(Q))

        var = []
        var_r = []
        for i, _Rsm_ in enumerate(Rsm):
            box_sm = smooth_box(box, R=_Rsm_, periodic=True).real
            var.append(np.std(box_sm.ravel())**2)

            if show_random:
                box_sm = smooth_box(box_r, R=_Rsm_, periodic=True).real
                var_r.append(np.std(box_sm.ravel())**2)

        ax.plot(Rsm, var, **kwargs)

        if show_random:
            norm = max(var) / max(var_r)
            ax.plot(Rsm, np.array(var_r) * norm, color='k', ls=':',
                label=r'noise w/ same $Q$ (re-scaled)')

        ax.set_xlabel(r'smoothing scale [pixels]')
        ax.set_ylabel(r'$\sigma^2$')
        ax.set_xscale('log')
        ax.legend()

        return ax

    def plot_Qint_vs_Q(self, Qarr=None, R=5., sigma=1., gamma=0.,
        Lbox=100., vox=1., seed=None, fig=1, ax=None,
        allow_partial_ionization=False, **kwargs): # pragma: no cover

        if ax is None:
            fig, ax = pl.subplots(1, 1, num=fig)

        if Qarr is None:
            Qarr = np.arange(0.1, 1.1, 0.2)

        pb = ProgressBar(Qarr.size, name="ps(bb;Q)")
        pb.start()

        Qint = []
        Qint_num = []
        colors = ['k', 'b', 'm', 'c', 'g', 'y', 'orange', 'r'] * 10
        for i, _Q_ in enumerate(Qarr):
            pb.update(i)

            box, box_tot = self.get_box_bubbles(z=np.inf, Lbox=Lbox, vox=vox,
                Q=_Q_, R=R, sigma=sigma, gamma=gamma, use_kdtree=True, seed=seed,
                allow_partial_ionization=allow_partial_ionization)
            Npix = box.shape[0]

            Qint.append(self.get_Qint(d=np.inf, Q=_Q_, R=R, sigma=sigma,
                gamma=gamma))

            Qint_box = box_tot[box_tot > 1].size / float(box_tot.size)
            Qint_num.append(Qint_box)

        ax.plot(Qarr, Qint, color='k', ls='--', label='analytic')
        ax.plot(Qarr, Qint_num, label='numerical', **kwargs)
        ax.set_xlabel(r'$Q$')
        ax.set_ylabel(r'$Q_{\rm{int}}$')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_xticks(np.arange(0.05, 1.1, 0.1), minor=True)
        ax.set_yticks(np.arange(0.05, 1.05, 0.1), minor=True)
        ax.legend()

        return ax

    def plot_Q(self, Qarr=None, Rsm=1, R=5., sigma=1., gamma=0.,
        Lbox=100., vox=1., seed=None, fig=1, ax=None,
        allow_partial_ionization=False, **kwargs): # pragma: no cover

        if ax is None:
            fig, ax = pl.subplots(1, 1, num=fig)

        if Qarr is None:
            Qarr = np.arange(0.1, 1.1, 0.2)

        pb = ProgressBar(Qarr.size, name="ps(bb;Q)")
        pb.start()

        for i, _Q_ in enumerate(Qarr):
            pb.update(i)

            box, box_tot = self.get_box_bubbles(z=np.inf, Lbox=Lbox, vox=vox,
                seed=seed, Q=_Q_, R=R, sigma=sigma, gamma=gamma,
                allow_partial_ionization=allow_partial_ionization)

            Qbox = box[box == 0].size / float(box.size)

            ax.scatter(_Q_, Qbox, **kwargs)

        pb.finish()

        ax.plot(Qarr, Qarr, color='k', ls=':')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_xticks(np.arange(0.05, 1.1, 0.1), minor=True)
        ax.set_yticks(np.arange(0.05, 1.05, 0.1), minor=True)
        ax.legend()

        ax.set_xlabel(r'$Q_{\rm{in}}$')
        ax.set_ylabel(r'$Q_{\rm{box}}$')

        return ax

    def plot_Pbb_vs_Q(self, Qarr=None, karr=None, R=5., sigma=1., gamma=0.0,
        Lbox=100., vox=1., seed=None, fig=1, ax=None): # pragma: no cover
        """
        Compare power spectra of analytic model to that derived from a box.
        """

        if Qarr is None:
            Qarr = np.arange(0.1, 1.1, 0.2)

        if ax is None:
            fig, axes = pl.subplots(1, len(Qarr), figsize=(4*len(Qarr), 4))
        else:
            axes = [ax]

        if karr is None:
            logkarr = np.arange(-2, 1.1, 0.1)
            logkarr_e = bin_c2e(logkarr)
            kbins = 10**logkarr_e
            karr = 10**logkarr

        pb = ProgressBar(Qarr.size, name="ps(bb;Q)")
        pb.start()

        colors = ['k', 'b', 'm', 'c', 'g', 'y', 'orange', 'r'] * 10
        for i, _Q_ in enumerate(Qarr):
            pb.update(i)

            box, box_tot = self.get_box_bubbles(z=np.inf, Lbox=Lbox, vox=vox,
                seed=seed, Q=_Q_, R=R, sigma=sigma, gamma=gamma)

            ps_num, k_num = pbox.get_power(box, Lbox, ignore_zero_mode=1,
                bins=kbins)

            axes[i].loglog(k_num, k_num**3 * ps_num / 2. / np.pi**2, color=colors[i],
                label=r'$Q=%.1f$' % _Q_)

            ps = self.get_ps_bb(z=np.inf, k=karr, Q=_Q_, R=R, sigma=sigma,
                gamma=gamma)

            axes[i].loglog(karr, karr**3 * ps / 2. / np.pi**2, color=colors[i],
                ls='--')

        pb.finish()

        for ax in axes:
            ax.set_xlabel(r'$k$')
            ax.legend(loc='upper left')
            ax.set_ylim(1e-3, 2e-1)

        axes[0].set_ylabel(r'$\Delta_{bb}^2(k)$')

        return axes
