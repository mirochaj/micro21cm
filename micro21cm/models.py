"""

models.py

Authors: Jordan Mirocha and Julian B. Munoz
Affiliation: McGill University and Harvard-Smithsonian Center for Astrophysics
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import os
import numpy as np
from scipy.optimize import fmin, fsolve
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, quad
from scipy.special import erfcinv, erf, erfc
from .util import get_cf_from_ps, get_ps_from_cf, ProgressBar, \
    CTfit, Tgadiabaticfit

tiny_Q = 1e-3
tiny_cf = 1e-16

try:
    import camb
except ImportError:
    pass

try:
    import h5py
except ImportError:
    pass

PATH = os.environ.get("MICRO21CM")

class BubbleModel(object):
    def __init__(self, bubbles=True, bubbles_ion=True, bubbles_pdf='lognormal',
        bubbles_via_Rpeak=True, bubbles_model='fzh04',
        include_adiabatic_fluctuations=True,
        include_P1=True, include_P2=True,
        include_P1_corr=False, include_P2_corr=False, include_overlap_corr=0,
        include_cross_terms=1, include_rsd=2, include_mu_gt=-1.,
        use_volume_match=1, density_pdf='lognormal',
        Rmin=1e-2, Rmax=1e3, NR=1000,
        omega_b=0.0486, little_h=0.67, omega_m=0.3089, ns=0.96,
        transfer_kmax=500., transfer_k_per_logint=11, zmax=20.,
        use_pbar=False, approx_linear=True, **_kw_):
        """
        Make a simple bubble model of the IGM.

        Main Parameters
        ---------------
        bubbles : int, bool
            If False, just include density fluctuations and allow mean
            IGM spin temperature to vary as free parameter. If True,
            include bubbles (ionized or heated; see `bubbles_ion` below),
            that fill some fraction of the volume Q and follow some
            size distribution `bubbles_pdf`.
        bubbles_ion : int, bool
            If True, assumes b=1 refers to ionization, if False, assumes b=1
            corresponds to full heating.
        bubbles_pdf : str
            Bubble size distribution. Will get normalized to volume-filling
            fraction Q automatically. Current options: 'lognormal',
            'plexp', or a user-defined option, which requires that the user supply
            an array of radii and an unnormalized BSD as a tuple. The two parameters
            R and sigma characterize the PDF, and are the avg and rms of radii for
            'lognormal'. For 'plexp' R is the typical size, and gamma is a power-
            law index for bubbles with radii < R.
        bubbles_via_Rpeak : bool
            By default, the parameter 'R' throughput refers to where the BSD
            peaks, where BSD is dn/dR. To work in terms of the peak in the
            volume-weighted BSD, R**3 * dn/dlnR, set bubbles_via_Rpeak=True.
            Then, all R values supplied to class methods will be converted to
            the scale where dn/dR peaks internally before calling get_bsd.
        include_adiabatic_fluctuations : bool
            If True, inclue a correction factor that accounts for the fact
            that density and kinetic temperature are correlated. Uses
            fitting formulae from Muñoz et al. (2015). See Secton 2.2 in
            Mirocha, Munoz et al. (2021) for details.
        include_rsd : int, bool
            Include simple treatment of redshift space distortions?
        include_mu_gt : float
            If include_rsd > 0, this sets the lower-bound of the integral
            that averages the 3-D power spectrum over \mu.
        include_P1_corr : bool
            If True, apply "kludge" to one bubble terms in order to guarantee
            that fluctuations vanish as Q -> 1.
        include_cross_terms : bool, int
            If > 0, will allow terms involving both ionization and density to
            be non-zero. See Section 2.4 in Mirocha, Munoz et al. (2021) for
            details.
        density_pdf : str
            Sets functional form of 1-D PDF of density field. Only options
            are normal and lognormal. Affects computation of cross-terms
            (see previous parameter).
        use_volume_match : bool, int
            Determines method used to find characteristic density of ionized
            bubbles, if include_cross_terms > 0. Should probably just use
            a value of 1, in which case we can deprecate this parameter.
            Currently it just determines what scale is used to smooth the
            density field.
        Rmin, Rmax: float
            Bounds of configuration space distance array, self.tab_R. This
            is used for the computation of all correlation functions and
            real-space ensemble averages. Units: cMpc / h.
        NR : int
            Number of grid points to use between Rmin and Rmax.


        Cosmology
        ---------


        Transfer Functions
        ------------------

        """

        # Check for user-defined BSD.
        if type(bubbles_pdf) == str:
            self.bubbles_pdf = bubbles_pdf
            self.Rmin = Rmin
            self.Rmax = Rmax
            self.NR = int(NR)
        else:
            self._tab_user_R, self._tab_user_bsd = bubbles_pdf
            assert self._tab_user_bsd.shape == self._tab_user_R.shape
            self.bubbles_pdf = 'user'
            self.Rmin = self._tab_user_R.min()
            self.Rmax = self._tab_user_R.max()
            self.NR = self._tab_user_R.size

        self.bubbles = bubbles
        self.bubbles_ion = bubbles_ion
        self.bubbles_via_Rpeak = bubbles_via_Rpeak
        self.use_pbar = use_pbar
        self.bubbles_model = bubbles_model
        self.include_P1_corr = include_P1_corr
        self.include_P2_corr = include_P2_corr
        self.include_overlap_corr = include_overlap_corr
        self.include_P1 = include_P1
        self.include_P2 = include_P2
        self.include_adiabatic_fluctuations = \
            include_adiabatic_fluctuations
        self.include_cross_terms = include_cross_terms
        self.include_rsd = include_rsd
        self.include_mu_gt = include_mu_gt
        self.use_volume_match = use_volume_match
        self.approx_linear = approx_linear
        self.density_pdf = density_pdf

        self.params = ['Ts', 'Q']
        if self.bubbles:
            self.params.append('R')

            if self.bubbles_pdf == 'lognormal':
                self.params.append('sigma')
            elif self.bubbles_pdf == 'normal':
                self.params.extend(['sigma'])
            elif self.bubbles_pdf == 'plexp':
                self.params.append('gamma')
            elif self.bubbles_pdf == 'user':
                pass
            else:
                raise NotImplemented('Dunno BSD {}'.format(self.bubbles_pdf))

        self.transfer_params = \
            {
             'k_per_logint': transfer_k_per_logint,
             'kmax': transfer_kmax,
             'extrap_kmax': True,
             'zmax': zmax,
            }

        omega_cdm = omega_m - omega_b

        self.cosmo_params = \
            {
             'H0': little_h * 100.,
             'ombh2': omega_b * little_h**2,
             'omch2': omega_cdm * little_h**2,
             'w': -1,
             'lmax': 10000,
            }

    def _init_cosmology(self):

        transfer_pars = camb.model.TransferParams(**self.transfer_params)

        # Should setup the cosmology more carefully.
        self._cosmo_ = camb.set_params(WantTransfer=True,
            Transfer=transfer_pars, **self.cosmo_params)

        if self.approx_linear:
            nonlin = camb.model.NonLinear_none
        else:
            nonlin = camb.model.NonLinearModel

        #self._cosmo_.set_matter_power(redshifts=self._redshifts,
        #    nonlinear=nonlin)

        # `P` method of `matter_ps` is function of (z, k)
        self._matter_ps_ = camb.get_matter_power_interpolator(self._cosmo_,
            nonlinear=self._cosmo_.NonLinear, **self.transfer_params)

    @property
    def cosmo(self):
        if not hasattr(self, '_cosmo_'):
            self._init_cosmology()
        return self._cosmo_

    def get_Tcmb(self, z):
        """ Return the CMB temperature at redshift `z` in K."""
        return self.cosmo.TCMB * (1. + z)

    def get_Tgas(self, z):
        """
        Return the gas temperature of an unheated IGM in K.

        .. note :: Uses fit to results of recombination code. Just pull
            from CAMB.

        """
        return Tgadiabaticfit(z)

    def get_alpha(self, z, Ts):
        """
        Return value of alpha, the parameter the controls whether bubbles
        are assumed to be fully ionized or fully heated.
        """
        if not self.bubbles:
            return 0.
        elif self.bubbles_ion:
            return -1
        else:
            return self.get_Tcmb(z) / (Ts - self.get_Tcmb(z))

    def get_ps_mm(self, z, k, **_kw_):
        """ Just a wrapper around `get_ps_matter`. """
        return self.get_ps_matter(z, k)

    def get_ps_matter(self, z, k, **_kw_):
        """
        Return matter power spectrum, P(k), at redshift `z` on scale `k`.
        """
        if not hasattr(self, '_matter_ps_'):
            self._init_cosmology()

        #self.cosmo.set_matter_power(redshifts=[z], kmax=k.max())
        #results = camb.get_results(self.cosmo)

        #kh, z, pk = results.get_matter_power_spectrum(minkh=k.min(), maxkh=k.max(), npoints=k.size)

        return self._matter_ps_.P(z, k)

    def get_dTb_avg(self, z, Q=0.0, R=5., sigma=0.5, gamma=0.,
        alpha=0., Ts=np.inf):
        """
        Return volume-averaged 21-cm brightness temperature, i.e., the
        global 21-cm signal.

        .. note :: This is different from `get_dTb_bulk` (see next function)
            because appropriately weights by volume, and accounts for
            cross-correlations between ionization and density.

        """
        bd = self.get_bubble_density(z, Q=Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha) * Q
        return self.get_dTb_bulk(z, Ts=Ts) * (1. - Q - bd)

    def get_dTb_bulk(self, z, Ts=np.inf):
        """
        Differential brightness temperature in "bulk" IGM.
        """
        return 27. * (self.cosmo.ombh2 / 0.023) * \
            np.sqrt(0.15 * (1.0 + z) / (self.cosmo.omch2 + self.cosmo.ombh2) \
            / 10.) * (1.0 - self.get_Tcmb(z) / Ts)

    @property
    def tab_R(self):
        if not hasattr(self, '_tab_R'):
            if self.bubbles_pdf == 'user':
                self._tab_R = self._tab_user_R
            else:
                self._tab_R = np.logspace(np.log10(self.Rmin),
                    np.log10(self.Rmax), self.NR)
        return self._tab_R

    def _get_bsd_unnormalized(self, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0.):
        """
        Return an unnormalized version of the bubble size distribution.

        .. note :: This is dn/dR! Not dn/dlogR. Multiply by
            self.tab_R to obtain the latter.

        """

        if self.bubbles_pdf == 'user':
            return self._tab_user_bsd

        logRarr = np.log(self.tab_R)
        logR = np.log(R)
        if self.bubbles_pdf == 'lognormal':
            bsd = np.exp(-(logRarr - logR)**2 / 2. / sigma**2) \
                / self.tab_R / sigma / np.sqrt(2 * np.pi)
            if alpha != 0:
                bsd *= (1. + erf(alpha * (logRarr - logR) / sigma / np.sqrt(2.)))
        elif self.bubbles_pdf in ['normal', 'gaussian']:
            bsd = np.exp(-(self.tab_R - R)**2 / 2. / sigma**2) \
                / sigma / np.sqrt(2 * np.pi)
            if alpha != 0:
                bsd *= (1. + erf(alpha * (self.tab_R - R) / sigma / np.sqrt(2.)))
        elif self.bubbles_pdf == 'plexp':
            bsd = (self.tab_R / R)**gamma * np.exp(-self.tab_R / R)
        else:
            raise NotImplemented("Unrecognized `bubbles_pdf`: {}".format(
                self.bubbles_pdf))

        return bsd

    def _cache_bsd(self, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0.):

        if not hasattr(self, '_cache_bsd_'):
            self._cache_bsd_ = {}

        key = (Q, R, sigma, gamma, alpha)
        if key in self._cache_bsd_:
            return self._cache_bsd_[key]

        return None

    def get_bsd(self, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0., **_kw_):
        """
        Compute the bubble size distribution (BSD).

        .. note :: This is dn/dR! Not dn/dlogR. Multiply by
            self.tab_R to obtain the latter.

        This is normalized so that:

        1 - \exp{-\int dn/dlnR(R) V(R) dlnR} = Q

        Parameters
        ----------
        Q : int, float
            Fraction of volume filled by bubbles. Normalizes the BSD.
        R : int, float
            Typical bubble size [cMpc / h]. Note that this is the location of
            the peak in dn/dlnR!
        sigma : int, float
            For `lognormal` BSD, this characterizes the width of the
            distribution.
        gamma : int, float
            For `plexp` BSD this is the power-law slope.

        """

        #cached_bsd = self._cache_bsd(Q, R, sigma, gamma, alpha)
        #if cached_bsd is not None:
        #    return cached_bsd

        # In this case, assumes user input is actually peak in V dn/dlogR,
        # convert to peak in dn/dR before calling _get_bsd_unnormalized
        if self.bubbles_pdf == 'user':
            pass
        else:
            if self.bubbles_via_Rpeak:
                _R = R * 1
                R = self.get_R_from_Rpeak(Q=Q, R=R, sigma=sigma, gamma=gamma)

        # Should cache bsd too.
        _bsd = self._get_bsd_unnormalized(Q=Q, R=R, sigma=sigma,
            gamma=gamma, alpha=alpha)

        # _bsd here is dn/dR, will multiple by R to obtain dn/dlnR
        # before integrating over V(R)
        V = 4 * np.pi * self.tab_R**3 / 3.
        integ = _bsd * self.tab_R * V
        norm = 1. / integ.max()
        integ = np.trapz(integ * norm, x=np.log(self.tab_R)) / norm

        corr = -1 * np.log(1 - Q) / integ

        # Normalize to provided ionized fraction
        bsd = _bsd * corr

        #self._cache_bsd_[(Q, R, sigma, gamma, alpha)] = bsd

        return bsd

    def get_Rpeak(self, Q=0., sigma=0.5, R=5., gamma=0., alpha=0.,
        assume_dndlnR=True, **_kw_):
        if self.bubbles_via_Rpeak:
            return R
        else:
            return self.get_Rpeak_from_R(Q=Q, R=R, sigma=sigma, gamma=gamma,
                assume_dndlnR=assume_dndlnR)

    def get_Rpeak_from_R(self, Q=0., sigma=0.5, R=5., gamma=0., alpha=0.,
        assume_dndlnR=True, **_kw_):
        """
        Return scale at which volume-weighted BSD peaks.

        Parameters
        ----------
        assume_dndlnR : bool
            If True, will return scale at dn/dlnR peaks. If False, will instead
            return scale corresponding to peak in dn/dR.
        """

        if self.bubbles_pdf == 'lognormal':
            if assume_dndlnR:
                Rp = R * np.exp(3 * sigma**2)
            else:
                Rp = np.exp(np.log(R) - sigma**2)

        elif self.bubbles_pdf in ['normal', 'gaussian']:
            Rp = 0.
        elif self.bubbles_pdf == 'plexp':
            if assume_dndlnR:
                Rp = R * (4. + gamma)
            else:
                Rp = R * (3. + gamma)
        else:
            raise NotImplemented("Unrecognized `bubbles_pdf`: {}".format(
                self.bubbles_pdf))

        return Rp

    def get_R_from_Rpeak(self, Q=0., sigma=0.5, R=5., gamma=0., alpha=0.,
        assume_dndlnR=True, **_kw_):
        """
        Return scale at which volume-weighted BSD peaks.

        Parameters
        ----------
        assume_dndlnR : bool
            If True, will return scale at dn/dlnR peaks. If False, will instead
            return scale corresponding to peak in dn/dR.
        """

        if self.bubbles_pdf == 'lognormal':
            if assume_dndlnR:
                Rp = R * np.exp(-3 * sigma**2)
            else:
                Rp = np.exp(np.log(R) + sigma**2)

        elif self.bubbles_pdf in ['normal', 'gaussian']:
            Rp = 0.
        elif self.bubbles_pdf == 'plexp':
            if assume_dndlnR:
                Rp = R / (4. + gamma)
            else:
                Rp = R / (3. + gamma)
        else:
            raise NotImplemented("Unrecognized `bubbles_pdf`: {}".format(
                self.bubbles_pdf))

        return Rp

    def get_bsd_cdf(self, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0.):
        """
        Compute the cumulative distribution function for the bubble size dist.
        """

        pdf = self.get_bsd(Q=Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha)
        cdf = cumtrapz(pdf * self.tab_R, x=np.log(self.tab_R), initial=0.0)

        return cdf / cdf[-1]

    def get_nb(self, Q=0.0, R=5., sigma=0.5, gamma=0.0, alpha=0.):
        """
        Compute the number density of bubbles [(h / Mpc)^3].
        """
        pdf = self.get_bsd(Q=Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha)
        return np.trapz(pdf * self.tab_R, x=np.log(self.tab_R))

    def get_overlap_corr(self, d, Q=0.0, R=5., sigma=0.5, gamma=0.,
        alpha=0.0, exclusion=0, method=0, which_vol='o'): # pragma: no cover

        if 0 < method < 1:
            suppression = method
        elif method % 1 != 0:
            suppression = method % int(method)
        else:
            suppression = 1

        if (not exclusion):
            P1e = self.get_P1(d, Q=Q, R=R, sigma=sigma, gamma=gamma,
                alpha=alpha, exclusion=1, use_corr=False)
            #P2 = self.get_P2(d, Q=Q, R=R, sigma=sigma, gamma=gamma,
            #    alpha=alpha)

            _Q_ = Q

            # Average between two corrections we could use or just use Q.
            if int(method) == 1:
                corr = (1. - Q)
            elif int(method) == 2:
                corr = (1. - np.sqrt(_Q_ * P1e))
            elif int(method) >= 3:
                Qrelevant = self.get_intersectional_vol(d, Q=Q, R=R, sigma=sigma,
                    gamma=gamma, alpha=alpha, which_vol=which_vol)

                if method == 3:
                    corr = np.exp(-Q)
                elif method == 4:
                    corr = np.exp(-np.minimum(Qrelevant, Q))
                elif method == 5:
                    Qtot = self.get_Qtot(Q=Q, R=R, sigma=sigma, gamma=gamma,
                        alpha=alpha)
                    corr = np.exp(-Qrelevant/Qtot)
                elif method == 6:
                    corr = 1. - np.minimum(Qrelevant, Q)
                elif method == 7:
                    Qtot = self.get_Qtot(Q=Q, R=R, sigma=sigma, gamma=gamma,
                        alpha=alpha)
                    corr = np.exp(-np.minimum(Qrelevant, Q)/Q)
                else:
                    raise NotImplemented('help')
            else:
                raise NotImplemented('help')

        else:
            corr = 1.

        return np.minimum(corr, 1.) * suppression

    def get_intersectional_vol(self, d, Q=0., R=5., sigma=0.5, gamma=0.,
        alpha=0., which_vol='o', **_kw_): # pragma: no cover

        bsd = self.get_bsd(Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha)

        V_R = 4. * np.pi * self.tab_R**3 / 3.
        V_o = self.get_overlap_vol_arr(d)

        if which_vol == 'o':
            V = V_o
        elif which_vol == 'x':
            V = V_R - V_o
        elif which_vol == 'tot':
            V = V_R

        integ1 = np.trapz(bsd * V * self.tab_R, x=np.log(self.tab_R))
        integ2 = np.trapz(bsd * V * self.tab_R, x=np.log(self.tab_R))

        #Pdub = (1. - np.exp(-integ1))**2 * np.exp(-integ1) / 2.

        return integ1 - (1. - np.exp(-integ2))

    def get_Qint(self, d, Q=0., R=5., sigma=0.5, gamma=0.,
        alpha=0., which_vol='tot', **_kw_): # pragma: no cover
        return self.get_intersectional_vol(d, Q=Q, R=R, sigma=sigma,
            gamma=gamma, alpha=alpha, which_vol=which_vol, **_kw_)

    def get_Qtot(self, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0.0): # pragma: no cover
        bsd = self.get_bsd(Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha)

        V = 4. * np.pi * self.tab_R**3 / 3.
        integ = np.trapz(bsd * V * self.tab_R, x=np.log(self.tab_R))
        return integ

    def get_P1(self, d, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0.0,
        exclusion=0, use_corr=True):
        """
        Compute 1 bubble term.
        """

        if not self.include_P1:
            return 0.0

        bsd = self.get_bsd(Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha)
        V_o = self.get_overlap_vol_arr(d)

        if exclusion:
            V = 4. * np.pi * self.tab_R**3 / 3.
            integ = np.trapz(bsd * (V - V_o) * self.tab_R,
                x=np.log(self.tab_R))
        else:
            integ = np.trapz(bsd * V_o * self.tab_R,
                x=np.log(self.tab_R))

        if self.bubbles_model == 'fzh04':
            P1 = 1. - np.exp(-integ)
        elif self.bubbles_model == 'test':
            P1 = integ * np.exp(-integ)

        if use_corr and self.include_P1_corr:
            corr = self.get_overlap_corr(d, Q=Q, R=R, sigma=sigma,
                gamma=gamma, alpha=alpha, method=self.include_P1_corr,
                which_vol='tot')
        else:
            corr = 1.

        return P1 * corr

    def get_P2(self, d, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0., xi_bb=0.,
        use_corr=True):
        """
        Compute 2 bubble term.
        """

        if not self.include_P2:
            return Q**2

        bsd = self.get_bsd(Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha)
        V_o = self.get_overlap_vol_arr(d)

        V = 4. * np.pi * self.tab_R**3 / 3.

        integ1 = np.trapz(bsd * (V - V_o) * self.tab_R,
            x=np.log(self.tab_R))
        integ2 = np.trapz(bsd * (V - V_o) * (1. + xi_bb) *
            self.tab_R, x=np.log(self.tab_R))

        if self.bubbles_model == 'fzh04':
            P2 = (1. - np.exp(-integ1)) * (1. - np.exp(-integ2))
        elif self.bubbles_model == 'test':
            #P2 = (integ1 * np.exp(-integ1))**2
            P2 = (1. - np.exp(-integ1)) * (1. - np.exp(-integ2))

        if use_corr and self.include_P2_corr:
            corr = self.get_overlap_corr(d, Q=Q, R=R, sigma=sigma,
                gamma=gamma, alpha=alpha, method=self.include_P2_corr,
                which_vol='x')

        else:
            corr = 1.

        return P2 * corr

    def get_PN(self, d, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0.,
        xi_bb=0., use_corr=True, N=1): # pragma: no cover
        """
        Experimental treatment of 'overlap' component of P2.
        """

        bsd = self.get_bsd(Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha)

        V_o = self.get_Vo_2d(d)
        V_R = 4. * np.pi * self.tab_R**3 / 3.

        P = np.zeros_like(self.tab_R)
        for i, R1 in enumerate(self.tab_R):

            # If no overlap, will have been counted in 2-bubble term.
            ok = V_o[i,:] > 0
            V1 = (4. * np.pi * R1**3 / 3.)

            # Two points in same structure
            if N == 1:
                _P = bsd[i] * np.trapz(ok * bsd * V1 * V_R * self.tab_R,
                    x=np.log(self.tab_R))

                P[i] = _P * np.exp(-_P)
            # Two points in different structures
            else:
                # Abundance of merged bubbles
                _P = bsd[i] * np.trapz(ok * bsd * (V1 + V_R) * self.tab_R,
                    x=np.log(self.tab_R))

                P[i] = _P * np.exp(-_P)

        PN = np.trapz(P * self.tab_R, x=np.log(self.tab_R))

        return PN**2

    @property
    def tab_Vo_2d(self):
        if not hasattr(self, '_tab_Vo_2d_'):
            fn = '{}/input/overlap_vol_log10R_{:.1f}_{:.1f}_N_{:.0f}_2D.hdf5'.format(
                PATH, np.log10(self.Rmin), np.log10(self.Rmax), self.NR)

            if not os.path.exists('{}'.format(fn)):
                self._tab_Vo_2d_ = np.zeros([self.NR, self.NR])
                for i, d in enumerate(self.tab_R):
                    tab_d = [self.get_overlap_vol(d, R) \
                            for j, R in enumerate(self.tab_R)]
                    self._tab_Vo_2d_[i,:] = np.array(tab_d)

            else:
                with h5py.File(fn, 'r') as f:
                    self._tab_Vo_2d_ = np.array(f[('Vo')])

        return self._tab_Vo_2d_

    @property
    def tab_Vo_3d(self): # pragma: no cover
        if not hasattr(self, '_tab_Vo_3d_'):
            fn = '{}/input/overlap_vol_log10R_{:.1f}_{:.1f}_N_{:.0f}_3D.hdf5'.format(
                PATH, np.log10(self.Rmin), np.log10(self.Rmax), self.NR)

            if not os.path.exists('{}'.format(fn)):
                raise IOError("No such file: {}/input/{}".format(PATH, fn))

            with h5py.File(fn, 'r') as f:
                self._tab_Vo_3d_ = np.array(f[('Vo')])

        return self._tab_Vo_3d_

    def get_Vo_2d(self, d):
        k = np.argmin(np.abs(d - self.tab_R))
        return self.tab_Vo_3d[k]

    def get_overlap_vol_arr(self, d):
        tab = self.tab_Vo_2d
        i = np.argmin(np.abs(d - self.tab_R))
        return tab[i,:]

    def get_overlap_vol(self, d, R):
        """
        Return overlap volume of two spheres of radius R separated by distance d.

        Parameters
        ----------
        d : int, float
            Separation in Mpc/h.
        R : int, float, np.ndarray
            Bubble size(s) in Mpc/h.


        """

        V_o = (4. * np.pi / 3.) * R**3 - np.pi * d * (R**2 - d**2 / 12.)

        if type(R) == np.ndarray:
            V_o[d >= 2 * R] = 0
        else:
            if d >= 2 * R:
                return 0.0

        return V_o

    def get_overlap_vol_generic(self, d, r1, r2):
        """
        Return overlap volume of two spheres of radius R1 and R2, which area separated by distance d.

        .. note :: This will reduce to `get_overlap_vol` if R1==R2.

        Parameters
        ----------
        d : int, float
            Separation in Mpc/h.
        R1, R2 : int, float, np.ndarray
            Bubble size(s) in Mpc/h.

        """

        if r2 >= r1:
            R1 = r1
            R2 = r2
        else:
            R1 = r2
            R2 = r1


        Vo = np.pi * (R2 + R1 - d)**2 \
            * (d**2 + 2. * d * R1 - 3. * R1**2 \
             + 2. * d * R2 + 6. * R1 * R2 - 3. * R2**2) / 12. / d

        if type(Vo) == np.ndarray:
            # Small-scale vs. large Scale
            SS = d <= R2 - R1
            LS = d >= R1 + R2

            Vo[LS == 1] = 0.0

            if type(R1) == np.ndarray:
                Vo[SS == 1] = 4. * np.pi * R1[SS == 1]**3 / 3.
            else:
                Vo[SS == 1] = 4. * np.pi * R1**3 / 3.
        else:
            if d <= (R2 - R1):
                return 4. * np.pi * R1**3 / 3.
            elif d >= (R1 + R2):
                return 0.0

        return Vo#np.maximum(Vo, 0)

    def get_bb(self, z, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0.,
        separate=False, xi_bb=None, **_kw_):
        """
        Comptute <bb'> following FZH04 model.

        .. note :: By default, this is equivalent to the joint probability
            that two points are ionized, often denoted <xx'>. If considering
            heated bubbles, it is instead the probability that two points are
            both heated.

        Parameters
        ----------
        z : int, float
            Redshift. Note that <bb'> doesn't actually depend on redshift, but
            we keep it here for consistency with other methods. May change.

        Parameters
        ----------
        Q : int, float
            Fraction of volume filled by bubbles. Normalizes the BSD.
        R : int, float
            Typical bubble size [cMpc / h]. Note that this is the location of
            the peak in dn/dlnR!
        sigma : int, float
            For `lognormal` BSD, this characterizes the width of the
            distribution.
        gamma : int, float
            For `plexp` BSD this is the power-law slope.
        xi_bb : int, float, np.ndarray
            Excess probability that a second point is ionized given that the
            first point is ionized.
        separate : bool
            If separate==True, will return one- and two-bubble terms separately
            as a tuple (P1, P2). If separate==False, just returns the sum
            of terms.

        """

        if Q == 0 or (not self.bubbles):
            if separate:
                return np.zeros_like(self.tab_R), np.zeros_like(self.tab_R)
            else:
                return np.zeros_like(self.tab_R)

        pb = ProgressBar(self.tab_R.size, use=self.use_pbar,
            name="<bb'>(z={})".format(z))
        pb.start()

        if xi_bb is None:
            xi_bb = np.zeros_like(self.tab_R)
        elif type(xi_bb) != np.ndarray:
            xi_bb = xi_bb * np.ones_like(self.tab_R)
        else:
            # Assumes it's the same for all bubbles
            assert xi_bb.size == self.tab_R.size

        P1 = np.zeros_like(self.tab_R)
        P2 = np.zeros_like(self.tab_R)
        P12 = np.zeros_like(self.tab_R)
        P22 = np.zeros_like(self.tab_R)
        for i, RR in enumerate(self.tab_R):
            pb.update(i)
            P1[i] = self.get_P1(RR, Q=Q, R=R, sigma=sigma, alpha=alpha,
                gamma=gamma)
            P2[i] = self.get_P2(RR, Q=Q, R=R, sigma=sigma, alpha=alpha,
                gamma=gamma, xi_bb=xi_bb[i])

            if self.bubbles_model == 'test':
                P12[i] = self.get_PN(RR, Q=Q, R=R, sigma=sigma, alpha=alpha,
                    gamma=gamma, xi_bb=xi_bb[i], N=1)
                #P22[i] = self.get_PN(RR, Q=Q, R=R, sigma=sigma, alpha=alpha,
                #    gamma=gamma, xi_bb=xi_bb[i], N=2)

        pb.finish()

        if separate:
            return P1, (1 - P1) * P2#, P12
        else:
            if self.bubbles_model == 'fzh04':
                return P1 + (1 - P1) * P2
            elif self.bubbles_model == 'test':
                return P1 * (1 - P12) + P2 \
                     + P12 + P22

    def get_bn(self, z, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0., **_kw_):

        P1e = [self.get_P1(RR, Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha,
            exclusion=1) for RR in self.tab_R]
        P1e = np.array(P1e)

        P1 = [self.get_P1(RR, Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha) \
            for RR in self.tab_R]
        P1 = np.array(P1)

        P_bn = (1 - P1) * P1e * (1. - Q)

        P_bb = self.get_bb(z, Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha)
        d_i = self.get_bubble_density(z, Q=Q, R=R, sigma=sigma,
            gamma=gamma, alpha=alpha)

        ceil = P_bb * 2 * (2. * d_i - 1) * (1 - Q) / d_i / Q
        #print(d_i, ceil, P_bn)
        return P_bn#np.minimum(P_bn, ceil)

    def get_dd(self, z, **_kw_):
        """
        Get the matter correlation function, equivalent to <dd'>.

        .. note :: Will cache for given redshift `z` to save time.

        .. note :: Acceptance of keyword arguments _kw_ is just so that
            every routine in this class won't crash if given, e.g., the
            spin temperature. That way we can create a dictionary of
            kwargs for a given model and pass it to any method.

        """

        if not hasattr(self, '_cache_dd_'):
            self._cache_dd_ = {}

        if z in self._cache_dd_:
            return self._cache_dd_[z]

        dd = get_cf_from_ps(self.tab_R, lambda kk: self.get_ps_matter(z, kk))

        self._cache_dd_[z] = dd
        return dd

    #def get_bd(self, z, Q=0.0, R=5., sigma=0.5, alpha=0., bbar=1,
    #    bhbar=1, **_kw_):
    #    """
    #    Get the cross correlation function between bubbles and density,
    #    equivalent to <bd'>.
    #    """
#
    #    dd = self.get_dd(z)
#
    #    Rdiffabs=np.abs(self.Rarr - R)
    #    Rindex = Rdiffabs.argmin()
#
#
    #    fact =  bbar * bhbar * Q * dd
#
    #    xbar=Q
    #    #true formula is (1-xbar) = exp(-Q)
#   #     xbar= 1 - np.exp(-Q)
#
#
    #    bd = (1-xbar) * fact
    #    bd[self.Rarr <  R] = (1-xbar) *(1.0 - np.exp(fact[Rindex]) )
    #    #R is characteristic bubble size
#
#
    #    return bd
#
    #def Rd(self, z, Q=0.0, R=5., sigma=0.5, gamma=0, alpha=0., bbar=1, bhbar=1):
    #    """
    #    Normalize the cross correlation
#
    #    """
#
    #    bd = self.get_bd(z, Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha,
    #        bbar=bbar, bhbar=bhbar)
    #    dd = self.get_dd(z)
    #    bb = self.get_bb(z, Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha)
#
    #    return (bd/np.sqrt(bb*dd))

    def get_variance(self, z, r, field, Q=0.0, Ts=np.inf, R=5., sigma=0.5,
        gamma=0., alpha=0., xi_bb=None, kmin=1e-5, kmax=1e5, dlogk=0.05,
        rtol=1e-5, atol=1e-5):
        """
        Compute the variance of some `field`.
        """

        # CAMB already sets of an interpolant for the matter PS, so
        # we can skip straight to integrating over window function.
        if field in ['matter', 'm', 'mm', 'dd', 'density']:
            Pofk = lambda k: self.get_ps_matter(z, k)

            var = self.get_variance_from_ps(Pofk, r, kmin=kmin, kmax=kmax,
                rtol=rtol, atol=atol)

            return var

        ##
        # Other fields require us to make the interpolant ourselves.
        if field in ['bb', 'bubbles', 'ionization', 'xx', 'x',  'ion']:
            ps = lambda k: self.get_ps_bb(z, k, Q=Q, R=R, sigma=sigma,
                gamma=gamma, alpha=alpha)
        elif field in ['21cm']:
            ps = lambda k: self.get_ps_21cm(z, k, Q=Q, R=R, sigma=sigma,
                gamma=gamma, alpha=alpha, Ts=Ts)
        else:
            raise NotImplemented('help')

        karr = 10**np.arange(np.log10(kmin), np.log10(kmax)+dlogk, dlogk)
        tab_ps = ps(karr)

        Pofk = interp1d(karr, tab_ps, kind='cubic', bounds_error=False,
            fill_value=0.0)

        var = self.get_variance_from_ps(Pofk, r, kmin=kmin, kmax=kmax,
            rtol=rtol, atol=atol)

        return var

    def get_variance_mm(self, z, r, kmin=1e-5, kmax=1e5, dlogk=0.05,
        rtol=1e-5, atol=1e-5):
        """
        Return the variance in the matter field at redshift `z` when
        smoothing on scale `r`.
        """

        return self.get_variance(z, r, 'mm', kmin=kmin, kmax=kmax, dlogk=dlogk,
            rtol=rtol, atol=atol)

    def get_variance_bb(self, z, r, Q=0.5, R=5., sigma=0.5, gamma=None,
        alpha=0.0, kmin=1e-5, kmax=1e5, dlogk=0.05, rtol=1e-5, atol=1e-5):
        """
        Return the variance in the ionization field at redshift `z` when
        smoothing on scale `r`.
        """

        return self.get_variance(z, r, 'bb', Q=Q, R=R, sigma=sigma,
            gamma=gamma, alpha=0.0, kmin=kmin, kmax=kmax, dlogk=dlogk,
            rtol=rtol, atol=atol)

    def get_variance_21cm(self, z, r, Q=0.5, R=5., Ts=np.inf, sigma=0.5,
        gamma=None, alpha=0.0, kmin=1e-5, kmax=1e5, dlogk=0.05, rtol=1e-5,
        atol=1e-5):
        """
        Return the variance in the ionization field at redshift `z` when
        smoothing on scale `r`.
        """

        return self.get_variance(z, r, '21cm', Q=Q, R=R, sigma=sigma,
            gamma=gamma, alpha=alpha, Ts=Ts, kmin=kmin, kmax=kmax, dlogk=dlogk,
            rtol=rtol, atol=atol)

    def get_variance_from_ps(self, ps, R, kmin=1e-5, kmax=1e5, rtol=1e-5,
        atol=1e-2):
        """
        Compute variance of generic field from input power spectrum.

        Parameters
        ----------
        ps : function
            Input power spectrum, assumed to be function of k only.
        R : int, float
            Smoothing scale in [1/k] units.

        Numerical Parameters
        --------------------
        kmin, kmax : float
            Bounds in k-space to consider in integral.
        atol, rtol: float
            Absolute and relative tolerances for integration, respectively.

        Returns
        -------
        Variance!

        """

        ikw = dict(epsrel=rtol, epsabs=atol, limit=10000, full_output=1)
        Pofk = ps

        Wofk = lambda k: 3 * (np.sin(k * R) - k * R * np.cos(k * R)) \
            / (k * R)**3

        integrand_full = lambda k: Pofk(k) * np.abs(Wofk(k)**2) \
            * 4. * np.pi * k**2 / (2. * np.pi)**3


        kcrit = 1. / R
        norm = 1. / integrand_full(kcrit)

        integrand_1 = lambda k: norm * Pofk(k) * 4. * np.pi * k**2 \
            * 3 / (k * R)**3 / (2. * np.pi)**3
        integrand_2 = lambda k: norm * Pofk(k) * 4. * np.pi * k**2 \
            * 3 * k * R / (k * R)**3 / (2. * np.pi)**3

        var = quad(integrand_full, kmin, kcrit, **ikw)[0]

        first_bit = quad(integrand_1, kcrit, kmax, weight='sin', wvar=R, **ikw)
        second_bit = quad(integrand_2, kcrit, kmax, weight='cos', wvar=R, **ikw)
        new = first_bit[0] - second_bit[0]

        var += new / norm

        return var

    def get_density_threshold(self, z, Q=0.0, R=5., sigma=0.5,
        gamma=0, alpha=0, **_kw_):
        """
        Use "volume matching" to determine density level above which
        gas is ionized.

        Parameters
        ----------
        self.use_volume_match : int
            This is set at instantiation, and controls the smoothing scale
            used to compute the variance in the density field. The main
            options are:

                1: smooth on scale where volume-weighted BSD V dn/dlogR
                   peaks.
                2: smooth on scale where volume-weighted BSD V dn/dR peaks.
                3: smooth on scale where dn/dR peaks.


        Returns
        -------
        Both the density of bubbles and sigma_R of the density field
        smoothed on the appropriate scale.

        """

        # Hack!
        if (Q < tiny_Q) or (Q == 1):
            return -1, 0.0

        if self.use_volume_match == 1:
            if self.bubbles_via_Rpeak:
                Rsm = R
            else:
                bsd = self.get_bsd(Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha)
                # convert to dn/dlogR
                bsd = bsd * self.tab_R
                # weight by volume
                bsd = bsd * 4. * np.pi * self.tab_R**3 / 3.
                # Doesn't matter here but OK.
                bsd = bsd / Q
                # find peak in V dn/dlnR
                Rsm = self.tab_R[np.argmax(bsd)]
        elif self.use_volume_match == 2: # pragma: no cover
            bsd = self.get_bsd(Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha)
            # weight by volume
            bsd = bsd * 4. * np.pi * self.tab_R**3 / 3.
            # find peak in V dn/dR
            Rsm = self.tab_R[np.argmax(bsd)]
        elif self.use_volume_match == 3: # pragma: no cover
            if self.bubbles_via_Rpeak:
                bsd = self.get_bsd(Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha)
                Rsm = self.tab_R[np.argmax(bsd)]
            else:
                Rsm = R
        elif int(self.use_volume_match) == 4: # pragma: no cover
            frac = self.use_volume_match % 4
            bb1, bb2 = self.get_bb(z, Q=Q, R=R, sigma=sigma, gamma=gamma,
                alpha=alpha, separate=True)
            bb = bb1 + bb2
            P1_frac = bb1 / bb
            Rsm = np.exp(np.interp(np.log(frac), np.log(P1_frac[-1::-1]),
                np.log(self.tab_R[-1::-1])))
        else:
            raise NotImplemented('help')

        var_R = self.get_variance_mm(z, r=Rsm)
        sig_R = np.sqrt(var_R)

        # Just changes meaning of what `x` and `w` are.
        # For density_pdf = 'normal' or 'Gaussian', x = delta, for log-normal,
        # x = log(1 + \delta)
        if self.density_pdf.lower() in ['normal', 'gaussian']:
            w = sig_R
        else:
            w = np.sqrt(np.log(var_R + 1.))

        # Eq. 33
        x_thresh = np.sqrt(2) * w * erfcinv(2 * Q)

        return x_thresh, w

    def get_bubble_density(self, z, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0,
        **_kw_):
        """
        Return mean density in ionized regions.
        """

        # Hack!
        if (Q < tiny_Q) or (Q == 1):
            return 0.0

        x_thresh, w = self.get_density_threshold(z, Q=Q, R=R,
            sigma=sigma, gamma=gamma, alpha=alpha, **_kw_)

        # Normalization factor
        norm = 0.5 * erfc(x_thresh / w / np.sqrt(2.))

        if self.density_pdf.lower() in ['normal', 'gaussian']:
            del_i = np.exp(-x_thresh**2 / 2. / w**2) * w \
                / np.sqrt(2 * np.pi)
        else:
            del_i = 0.5 * (-1. + np.exp(w**2 / 2.) \
                * (1. + erf((w**2 - x_thresh) / np.sqrt(2.) / w)) \
                + erf(x_thresh / np.sqrt(2.) / w))

        # Sanity checks: do numerically
        #norm = quad(lambda x: Pofx(x), x_thresh, np.inf,
        #    limit=100000)[0]
        #del_i = quad(lambda x: Pofx(x) * x, x_thresh, np.inf)[0]

        return del_i / norm

    def get_cross_terms(self, z, Q=0.0, Ts=np.inf, R=5., sigma=0.5,
        gamma=0., alpha=0., beta=1., delta_ion=0., separate=False,
        **_kw_):
        """
        Compute all terms that involve bubble field and density field.

        Parameters
        ----------
        separate : bool
            If True, will return each term separately. To unpack results, do,
            e.g.,

                bd, bbd, bdd, bbdd, bbd_1pt, bd_1pt = \
                    model.get_cross_terms(z, separate=True, **kwargs)

        """

        bb = self.get_bb(z, Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha)
        bn = self.get_bn(z, Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha)
        dd = self.get_dd(z)
        _alpha = self.get_alpha(z, Ts)
        #beta_d, beta_T, beta_mu, beta_mu_T = self.get_betas(z, Ts)
        beta_phi, beta_mu = self.get_betas(z, Ts)
        #beta_sq = (beta_mu**2 + beta_phi**2 + 2 * beta_mu * beta_phi)

        if not self.include_cross_terms:
            d_i = 0
        elif self.use_volume_match:
            d_i = self.get_bubble_density(z, Q=Q, R=R, sigma=sigma,
                gamma=gamma, alpha=alpha)
        else:
            d_i = delta_ion

        d_n = -d_i * Q / (1. - Q)

        # Currently neglects terms containing b and b' (other than <bb'>)
        if self.include_cross_terms == 0:
            bd = np.zeros_like(self.tab_R)
            bd_1pt = np.zeros_like(self.tab_R)
            bbd = np.zeros_like(self.tab_R)
            bdd = Q * dd
            bbdd = bb * dd
        elif self.include_cross_terms == 1:
            bd = d_i * bb + d_n * bn
            bd_1pt = np.zeros_like(self.tab_R)
            bbd = np.zeros_like(self.tab_R)
            bdd = Q * dd
            bbdd = Q**2 * dd
        elif self.include_cross_terms == 2:
            bd = d_i * bb + d_n * bn
            bd_1pt = d_i * Q

            bbd = d_i * bb
            bdd = d_i * d_i * bb + d_i * d_n * bn
            bbdd = bb * d_i**2
        elif self.include_cross_terms == 3:
            bd = d_i * bb + d_n * bn
            bd_1pt = bbd = bbdd = bdd = np.zeros_like(self.tab_R)
            #bdd = d_i * d_i * bb + d_i * d_n * bn
        else:
            raise NotImplemented('Only know include_cross_terms=1,2,3!')

        tot = 2 * _alpha * bd + 2 * _alpha**2 * bbd + 2 * _alpha * bdd \
            + _alpha**2 * bbdd \
            - 2 * _alpha**2 * Q * bd_1pt -_alpha**2 * bd_1pt**2

        if separate:
            return 2 * _alpha * bd, 2 * _alpha**2 * bbd, 2 * _alpha * bdd, \
                _alpha**2 * bbdd, \
                - 2 * _alpha**2 * Q * bd_1pt, -_alpha**2 * bd_1pt**2
        else:
            return 2 * _alpha * bd + 2 * _alpha**2 * bbd + 2 * _alpha * bdd \
                + _alpha**2 * bbdd \
                - 2 * _alpha**2 * Q * bd_1pt - _alpha**2 * bd_1pt**2

    def get_CT(self,z,Ts):
        # if self.bubbles:
        #     return 0.0 #we do not include it for the cases with bubbles, only for density (revisit)
        # else:
        return CTfit(z) * min(1.0, self.get_Tgas(z) / Ts)

    def get_contrast(self, z, Ts):
        return 1. - self.get_Tcmb(z) / Ts

    def get_beta_T(self, z, Ts):
        return self.get_CT(z, Ts)

    def get_beta_phi(self, z, Ts):
        if not self.include_adiabatic_fluctuations:
            return 0.0

        con = self.get_contrast(z, Ts)
        corr = Ts / self.get_Tcmb(z)
        CT = self.get_CT(z, Ts)
        return CT / con / corr

    def get_betas(self, z, Ts):
        beta_p = self.get_beta_phi(z, Ts)
        beta_T = self.get_beta_T(z, Ts)

        if self.include_rsd == 1:
            beta_mu_sq = self.get_rsd_boost_dd(self.include_mu_gt)
        elif self.include_rsd == 2:
            beta_mu_sq = self.get_rsd_boost_dd(self.include_mu_gt)
        elif self.include_rsd == 3:
            beta_mu_sq = 2.
        else:
            beta_mu_sq = 1.

        return beta_p, np.sqrt(beta_mu_sq)

    def get_cf_21cm(self, z, Q=0.0, Ts=np.inf, R=5., sigma=0.5, gamma=0.,
        alpha=0., xi_bb=None, delta_ion=0.):

        bb = 1 * self.get_bb(z, Q, R=R, sigma=sigma, alpha=alpha, gamma=gamma,
            xi_bb=xi_bb)
        dd = 1 * self.get_dd(z)
        _alpha = self.get_alpha(z, Ts)

        dTb = self.get_dTb_bulk(z, Ts=Ts)

        avg_term = _alpha**2 * Q**2

        if self.include_overlap_corr:
            corr = self.get_overlap_corr(0.0, Q=Q, R=R, sigma=sigma,
                gamma=gamma, alpha=alpha, method=self.include_overlap_corr,
                which_vol='tot')
        else:
            corr = 1.

        bb *= corr
        avg_term *= corr

        # Include correlations in density and temperature caused by
        # adiabatic expansion/contraction.
        beta_phi, beta_mu = self.get_betas(z, Ts)

        dd *= (beta_mu + beta_phi)**2

        cf_21 = bb * _alpha**2 + dd - avg_term

        bd, bbd, bdd, bbdd, bbd_1pt, bd_1pt = \
            self.get_cross_terms(z, Q=Q, Ts=Ts, R=R, sigma=sigma, gamma=gamma,
                alpha=alpha, delta_ion=delta_ion, separate=True)

        bd *= (beta_mu + beta_phi)
        #bbd *= (beta_mu + beta_phi)
        bdd *= (beta_mu + beta_phi)
        #bbdd *= (beta_mu + beta_phi)**2
        #bd_1pt *= (beta_mu + beta_phi)
        #bbd_1pt *= (beta_mu + beta_phi)

        cf_21 += bd + bbd + bdd + bbdd + bbd_1pt + bd_1pt

        return dTb**2 * cf_21

    def get_ps_bb(self, z, k, Q=0.5, R=5., sigma=0.5, gamma=None, alpha=0.,
        xi_bb=None, **_kw_):
        return self._get_ps_bx(z, k, Q=Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha, xi_bb=xi_bb, which_ps='bb', **_kw_)

    def get_ps_bd(self, z, k, Q=0.5, R=5., sigma=0.5, gamma=None, alpha=0.,
        xi_bb=None, Ts=np.inf, **_kw_):
        ps = self._get_ps_bx(z, k, Q=Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha, xi_bb=xi_bb, which_ps='bd', **_kw_)

        return ps / 2. / self.get_alpha(z, Ts)

    def _get_ps_bx(self, z, k, Q=0.5, R=5., sigma=0.5, gamma=None, alpha=0.,
        xi_bb=None, which_ps='bb', **_kw_):

        if which_ps == 'bb':
            jp = self.get_bb(z, Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha,
                xi_bb=xi_bb)
            avg = Q**2
        elif which_ps == 'bd':
            bd, bbd, bdd, bbdd, bbd_1pt, bd_1pt = \
                self.get_cross_terms(z, separate=True, Q=Q, R=R,
                    sigma=sigma, gamma=gamma, alpha=alpha, xi_bb=xi_bb, **_kw_)
            jp = bd

            d_i = self.get_bubble_density(z=z, Q=Q, R=R,
                sigma=sigma, gamma=gamma, alpha=alpha, xi_bb=xi_bb, **_kw_)
            avg = 0.0

        else:
            raise NotImplemented('help')

        if self.include_overlap_corr:
            corr = self.get_overlap_corr(0.0, Q=Q, R=R, sigma=sigma,
                gamma=gamma, alpha=alpha, method=self.include_overlap_corr,
                which_vol='tot')
        else:
            corr = 1.

        cf = (jp - avg) * corr

        # cf == 0 causes problems
        cf[cf == 0] = tiny_cf

        # Setup interpolant
        _fcf = interp1d(np.log(self.tab_R), cf, kind='cubic',
            bounds_error=False, fill_value=tiny_cf)
        f_cf = lambda RR: _fcf.__call__(np.log(RR))

        if type(k) != np.ndarray:
            k = np.array([k])

        ps = get_ps_from_cf(k, f_cf=f_cf,
            Rmin=self.tab_R.min(), Rmax=self.tab_R.max())

        return ps

    def get_ps_21cm(self, z, k, Q=0.0, Ts=np.inf, R=5., sigma=0.5,
        gamma=0., alpha=0., xi_bb=None, delta_ion=0.):

        # Much faster without bubbles -- just scale P_mm
        if (not self.bubbles) or (Q < tiny_Q):
            ps_mm = np.array([self.get_ps_matter(z, kk) for kk in k])
            beta_phi, beta_mu = self.get_betas(z, Ts)
            beta_sq = (beta_mu + beta_phi)**2
            Tavg = self.get_dTb_avg(z, Q=Q, Ts=Ts, R=R, sigma=sigma,
                gamma=gamma, alpha=alpha)

            ps_21 = Tavg**2 * ps_mm * beta_sq

        else:
            # In this case, if include_rsd==True, each term will carry
            # its own correction term, so we don't apply a correction
            # explicitly here as we do above in the Q=0 density-driven limit.
            cf_21 = self.get_cf_21cm(z, Q=Q, Ts=Ts, R=R,
                sigma=sigma, gamma=gamma, alpha=alpha, xi_bb=xi_bb,
                delta_ion=delta_ion)

            # Setup interpolant
            _fcf = interp1d(np.log(self.tab_R), cf_21, kind='cubic',
                bounds_error=False, fill_value=0.)
            f_cf = lambda RR: _fcf.__call__(np.log(RR))

            if type(k) != np.ndarray:
                k = np.array([k])

            ps_21 = get_ps_from_cf(k, f_cf=f_cf,
                Rmin=self.tab_R.min(), Rmax=self.tab_R.max())

        return ps_21

    def get_rsd_boost_dd(self, mu):
        # This is just \int_{\mu_{\min}}^1 d\mu (1 + \mu^2)^2
        mod = (1. - mu) + 2. * (1. - mu**3) / 3. + 0.2 * (1. - mu**5)
        # Full correction weighted by 1/(1 - mu)
        return mod / (1. - mu)

    def get_rsd_boost_d(self, mu):
        # This is just \int_{\mu_{\min}}^1 d\mu (1 + \mu^2)
        mod = (1. - mu) + 1. * (1. - mu**3) / 3.
        # Full correction weighted by 1/(1 - mu)
        return mod / (1. - mu)

    def get_rsd_int_mu2(self, mu):
        return (1. - self.include_mu_gt**3) / 3. / (1. - self.include_mu_gt)

    def calibrate_ps(self, k_in, Dsq_in, Q, z=None, Ts=None, which_ps='bb',
        free_norm=True, maxiter=100, xtol=1e-2, ftol=1e-3, use_log=True,
        R=None, sigma=None, gamma=None, Ts_guess=None):
        """
        Find the best-fit micro21cm representation of an input
        power spectrum (presumably from 21cmFAST).

        .. note :: Useful for taking a log-normal BSD PS (ionization
            or 21-cm) and calibrating a different BSD's parameters.

        .. note :: Just minimizing the sum of squared difference between the
            input spectrum and our model.

        .. note :: For maxiter=100, this will take about ~1 minute in general.

        Parameters
        ----------
        k_in : np.ndarray
            Array of modes for input PS.
        Dsq_in : np.ndarray
            Dimensionless PS of whatever we're calibrating to.
        Q : int, float
            Global ionized fraction of input model.
        which_ps : str
            Can provide 'bb' if fitting to ionization power spectra, or
            '21cm' if fitting to 21-cm PS.
        R : int, float
            If fitting 21-cm PS, can provide R (and sigma or gamma) to
            just fit for the spin temperature.

        """

        if which_ps == 'bb':
            func_ps = self.get_ps_bb
            z = np.inf
        elif which_ps == '21cm':
            func_ps = self.get_ps_21cm
            assert z is not None, "Must provide `z` for 21-cm PS!"

            if R is None:
                assert Ts is not None, "Must provide `Ts` for 21-cm PS!"

        else:
            raise NotImplemented('Help!')

        fitting_Ts = (R is not None) and \
            ((sigma is not None) or (gamma is not None))

        if free_norm:
            ps = lambda pars: pars[2] \
                * func_ps(z=z, k=k_in, Q=Q, Ts=Ts,
                R=10**pars[0], sigma=pars[1], gamma=pars[1])
        else:
            if fitting_Ts:
                Tcmb = self.get_Tcmb(z)

                ps = lambda pars: func_ps(z=z, k=k_in, Q=Q, Ts=10**pars[0],
                    R=R, sigma=sigma, gamma=gamma)

                assert free_norm == False
            else:
                ps = lambda pars: func_ps(z=z, k=k_in, Q=Q, Ts=Ts,
                    R=10**pars[0], sigma=pars[1], gamma=pars[1])

        Dsq = lambda pars: k_in**3 * ps(pars) / 2. / np.pi**2

        # Assume error scales as k
        #err = 0.1 * Dsq_in * (k_in / min(k_in))
        if use_log:
            func = lambda pars: np.sum((np.log10(Dsq(pars)) - \
                np.log10(Dsq_in))**2)
        else:
            func = lambda pars: np.sum((Dsq(pars) - Dsq_in)**2)

        _guess_ = 1.2 if self.bubbles_pdf == 'lognormal' else -2.5

        # Use some intuition on guesses
        if fitting_Ts:
            if Ts_guess is None:
                guess = [0.]
            else:
                guess = [Ts_guess]
        elif Q <= 0.2:
            guess = [-0.5, _guess_]
        elif Q < 0.5:
            guess = [0.5, _guess_]
        elif Q < 0.7:
            guess = [1., _guess_]
        elif Q < 0.9:
            guess = [1.2, _guess_]
        else:
            guess = [1.5, _guess_]

        if free_norm:
            guess.append(1.)

        popt = fmin(func, guess, maxiter=maxiter, xtol=xtol, ftol=ftol)
        #popt = fsolve(func, guess, maxfev=maxiter, xtol=ftol,
        #    factor=0.1)

        if fitting_Ts:
            popt2 = fsolve(func, [np.log10(Tcmb *1.1)], maxfev=maxiter,
                xtol=ftol, factor=0.1)
        else:
            popt2 = None

        if fitting_Ts:
            kw = {'Ts': 10**popt[0], 'Ts_hi': 10**popt2[0]}
        else:
            par = 'sigma' if self.bubbles_pdf == 'lognormal' else 'gamma'

            kw = {'R': 10**popt[0], par: popt[1]}
            if free_norm:
                kw['norm'] = popt[2]

        return kw
