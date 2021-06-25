"""

models.py

Authors: Jordan Mirocha and Julian B. Munoz
Affiliation: McGill University and Harvard-Smithsonian Center for Astrophysics
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import camb
import numpy as np
from scipy.optimize import fmin
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, quad
from scipy.special import erfcinv, erf, erfc
from .util import get_cf_from_ps, get_ps_from_cf, ProgressBar, \
    CTfit, Tgadiabaticfit

try:
    import powerbox as pbox
except ImportError:
    pass

tiny_Q = 1e-3

class BubbleModel(object):
    def __init__(self, bubbles=True, bubbles_ion=True,
        bubbles_pdf='lognormal', bubbles_Rfree=True,
        bubbles_via_Rpeak=True,
        include_adiabatic_fluctuations=True, include_P1_corr=True,
        include_cross_terms=1, include_rsd=2, include_mu_gt=-1.,
        use_volume_match=1, density_pdf='lognormal',
        Rmin=1e-2, Rmax=1e3, NR=1000,
        omega_b=0.0486, little_h=0.67, omega_m=0.3089, ns=0.96,
        transfer_kmax=500., transfer_k_per_logint=11, zmax=20.,
        use_pbar=False, approx_small_Q=False, approx_linear=True, **_kw_):
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
            fraction Q automatically. Current options: 'lognormal' and
            'plexp'. The two parameters R and sigma characterize the PDF,
            and are the avg and rms of radii for 'lognormal'. For 'plexp' R
            is the critical radius, and gamma is a power-law index for bubbles
            with radii < R.
        bubbles_via_Rpeak : bool
            By default, the parameter 'R' throughput refers to where the BSD
            peaks, where BSD is dn/dR. To work in terms of the peak in the
            volume-weighted BSD, R**3 * dn/dlnR, set bubbles_via_Rpeak=True.
            Then, all R values supplied to class methods will be converted to
            the scale where dn/dR peaks internally before calling get_bsd.
        bubbles_Rfree : bool
            If True, the characteristic bubble size, R, will be treated
            as the free parameter. This means that the bubble density will
            be determined automatically to obtain the requested Q. If False,
            the bubble density, n_b, will be the free parameter, and R
            will be determined iteratively to ensure the BSD integrates to Q.
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

        self.Rmin = Rmin
        self.Rmax = Rmax
        self.NR = int(NR)
        self.bubbles = bubbles
        self.bubbles_ion = bubbles_ion
        self.bubbles_pdf = bubbles_pdf
        self.bubbles_via_Rpeak = bubbles_via_Rpeak
        self.bubbles_Rfree = bubbles_Rfree
        self.use_pbar = use_pbar
        self.approx_small_Q = approx_small_Q
        self.include_P1_corr = include_P1_corr
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
            if self.bubbles_Rfree:
                self.params.append('R')
            else:
                self.params.append('n_b')

            if self.bubbles_pdf == 'lognormal':
                self.params.append('sigma')
            elif self.bubbles_pdf == 'normal':
                self.params.extend(['sigma'])
            elif self.bubbles_pdf == 'plexp':
                self.params.append('gamma')
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

    def get_ps_matter(self, z, k):
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
        alpha=0., n_b=None, Ts=np.inf):
        """
        Return volume-averaged 21-cm brightness temperature, i.e., the
        global 21-cm signal.

        .. note :: This is different from `get_dTb_bulk` (see next function)
            because appropriately weights by volume, and accounts for
            cross-correlations between ionization and density.

        """
        bd = self.get_bubble_density(z, Q=Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha, n_b=n_b) * Q
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
            self._tab_R = np.logspace(np.log10(self.Rmin), np.log10(self.Rmax),
                self.NR)
        return self._tab_R

    def _get_R_from_nb(self, Q=0.0, sigma=0.5, gamma=0., alpha=0.,
        n_b=None, Qtol=1e-6, maxiter=10000, **_kw_):
        """
        If self.bubbles_Rfree == False, it means the bubble
        abundance, n_b, is our free parameter. In this case, we
        must iteratively solve for the characteristic bubble size
        needed to guarantee that our BSD integrates to Q.
        """
        # Need to do this iteratively.

        if not hasattr(self, '_cache_R'):
            self._cache_R = {}

        # Cache
        if (Q, sigma, gamma, alpha, n_b, Qtol) in self._cache_R.keys():
            return self._cache_R[(Q, sigma, gamma, alpha, n_b, Qtol)]

        logRarr = np.log(self.tab_R)

        # Initial guess for bubble size
        logR = np.log(2.)
        R = np.exp(logR)
        logRhist = []

        logRstep = 0.1

        # Iterate until we achieve requested Q.
        ct = 0
        while ct < maxiter:
            logRhist.append(logR)

            _bsd = self._get_bsd_unnormalized(Q=Q, R=np.exp(logR),
                sigma=sigma, gamma=gamma, alpha=alpha, n_b=n_b)

            # Normalize bsd so we get requested `n_b`
            norm = n_b / np.trapz(_bsd * self.tab_R, x=np.log(self.tab_R))
            _bsd *= norm

            # Compute Q
            V = 4. * np.pi * self.tab_R**3 / 3.
            integ = np.trapz(_bsd * V * self.tab_R, x=np.log(self.tab_R))

            if self.approx_small_Q:
                _Q_ = integ
            else:
                _Q_ = 1. - np.exp(-integ)

            if abs(Q - _Q_) < Qtol:
                break

            # If we're just flip-flopping and not converging, make step
            # size smaller
            if ct % 4 == 0:
                if np.unique(logRhist[-4:]).size == 2:
                    logRstep *= 0.5

            # Update guess, try again
            if _Q_ > Q:
                logR -= logRstep
            else:
                logR += logRstep

            ct += 1

        # This shouldn't ever happen but throw a warning if we hit `maxiter`.
        if ct == maxiter:
            print("WARNING: maxiter={} reached for Q={}, n_b={}, sigma={}".format(
                maxiter, Q, n_b, sigma
            ))
            print("(actual Q={})".format(_Q_))

        # Little kludge to ensure we get Q exactly in LS limit? Dep. on Qtol.
        #bsd *= Q / _Q_

        # Once we find the right bubble size, cache and return.
        self._cache_R[(Q, sigma, gamma, alpha, n_b, Qtol)] = np.exp(logR)

        return self._cache_R[(Q, sigma, gamma, alpha, n_b, Qtol)]

    def _get_bsd_unnormalized(self, Q=0.0, R=5., sigma=0.5,
        gamma=0., alpha=0., n_b=None):
        """
        Return an unnormalized version of the bubble size distribution.

        .. note :: This is dn/dR! Not dn/dlogR. Multiply by
            self.tab_R to obtain the latter.

        """
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

    def _cache_bsd(self, Q=0.0, R=5., sigma=0.5, gamma=0.,
        alpha=0., n_b=None):

        if not hasattr(self, '_cache_bsd_'):
            self._cache_bsd_ = {}

        key = (Q, R, sigma, gamma, alpha, n_b)
        if key in self._cache_bsd_:
            return self._cache_bsd_[key]

        return None

    def get_bsd(self, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0.,
        n_b=None, **_kw_):
        """
        Compute the bubble size distribution (BSD).

        .. note :: This is dn/dR! Not dn/dlogR. Multiply by
            self.tab_R to obtain the latter.

        This is normalized so that:

        if self.approx_small_Q:
            \int dn/dlnR(R) V(R) dlnR = Q
        else:
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

        #cached_bsd = self._cache_bsd(Q, R, sigma, gamma, alpha, n_b)
        #if cached_bsd is not None:
        #    return cached_bsd

        # In this case, assumes user input is actually peak in V dn/dlogR,
        # convert to peak in dn/dR before calling _get_bsd_unnormalized
        if self.bubbles_via_Rpeak:
            _R = R * 1
            R = self.get_R_from_Rpeak(Q=Q, R=R, sigma=sigma, gamma=gamma,
                n_b=n_b)

        if not self.bubbles_Rfree:
            R = self._get_R_from_nb(Q=Q, sigma=sigma, gamma=gamma,
                alpha=alpha, n_b=n_b)

        # Should cache bsd too.
        _bsd = self._get_bsd_unnormalized(Q=Q, R=R, sigma=sigma,
            gamma=gamma, alpha=alpha, n_b=n_b)

        # _bsd here is dn/dR, will multiple by R to obtain dn/dlnR
        # before integrating over V(R)
        V = 4 * np.pi * self.tab_R**3 / 3.
        integ = _bsd * self.tab_R * V
        norm = 1. / integ.max()
        integ = np.trapz(integ * norm, x=np.log(self.tab_R)) / norm

        if self.approx_small_Q:
            corr = Q / integ
        else:
            # In this case, normalize n_b such that
            # 1 - \exp{-\int n_b V_b dR} = Q
            corr = -1 * np.log(1 - Q) / integ

        # Normalize to provided ionized fraction
        bsd = _bsd * corr

        #self._cache_bsd_[(Q, R, sigma, gamma, alpha, n_b)] = bsd

        return bsd

    def get_Rpeak(self, Q=0., sigma=0.5, R=5., gamma=0., alpha=0.,
        n_b=None, assume_dndlnR=True, **_kw_):
        if self.bubbles_via_Rpeak:
            return R
        else:
            return self.get_Rpeak_from_R(Q=Q, R=R, sigma=sigma, gamma=gamma,
                n_b=n_b, assume_dndlnR=assume_dndlnR)

    def get_Rpeak_from_R(self, Q=0., sigma=0.5, R=5., gamma=0., alpha=0.,
        n_b=None, assume_dndlnR=True, **_kw_):
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
        n_b=None, assume_dndlnR=True, **_kw_):
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

    def get_bsd_cdf(self, Q=0.0, R=5., sigma=0.5, gamma=0.,
        alpha=0., n_b=None):
        """
        Compute the cumulative distribution function for the bubble size dist.
        """

        pdf = self.get_bsd(Q=Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha, n_b=n_b)
        cdf = cumtrapz(pdf * self.tab_R, x=np.log(self.tab_R), initial=0.0)

        return cdf / cdf[-1]

    def get_nb(self, Q=0.0, R=5., sigma=0.5, gamma=0.0, alpha=0.,
        n_b=None):
        """
        Compute the number density of bubbles [(h / Mpc)^3].
        """
        pdf = self.get_bsd(Q=Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha, n_b=n_b)
        return np.trapz(pdf * self.tab_R, x=np.log(self.tab_R))

    def get_P1(self, d, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0.0,
        n_b=None, exclusion=0, use_corr=True):
        """
        Compute 1 bubble term.
        """

        bsd = self.get_bsd(Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha, n_b=n_b)
        V_o = self.get_overlap_vol(self.tab_R, d)

        if exclusion:
            V = 4. * np.pi * self.tab_R**3 / 3.
            integ = np.trapz(bsd * (V - V_o) * self.tab_R,
                x=np.log(self.tab_R))
        else:
            integ = np.trapz(bsd * V_o * self.tab_R,
                x=np.log(self.tab_R))

        if self.approx_small_Q:
            P1 = integ
        else:
            P1 = 1. - np.exp(-integ)

        if self.include_P1_corr and use_corr and (not exclusion):
            P1e = self.get_P1(d, Q=Q, R=R, sigma=sigma, gamma=gamma,
                alpha=alpha, n_b=n_b, exclusion=1, use_corr=False)
            #P2 = self.get_P2(d, Q=Q, R=R, sigma=sigma, gamma=gamma,
            #    alpha=alpha, n_b=n_b)

            if self.approx_small_Q:
                _Q_ = Q
            else:
                _Q_ = 1. - np.exp(-Q)

            # Average between two corrections we could use or just use Q.
            if self.include_P1_corr == 2:
                corr = np.sqrt(_Q_ * P1e)
            else:
                corr = Q

            P1 *= (1. - corr)

        return P1

    def get_P2(self, d, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0.,
        n_b=None, xi_bb=0.):
        """
        Compute 2 bubble term.
        """

        if not self.approx_small_Q:
            return Q**2

        bsd = self.get_bsd(Q, R=R, sigma=sigma, gamma=gamma,
            alpha=alpha, n_b=n_b)
        V_o = self.get_overlap_vol(self.tab_R, d)

        V = 4. * np.pi * self.tab_R**3 / 3.

        integ1 = np.trapz(bsd * (V - V_o) * self.tab_R,
            x=np.log(self.tab_R))
        integ2 = np.trapz(bsd * (V - V_o) * (1. + xi_bb) *
            self.tab_R, x=np.log(self.tab_R))

        if self.approx_small_Q:
            return integ1 * integ2
        else:
            return (1. - np.exp(-integ1)) * (1. - np.exp(-integ2))

    def get_overlap_vol(self, R, d):
        """
        Return overlap volume of two spheres of radius R separated by distance d.

        Parameters
        ----------
        R : int, float, np.ndarray
            Bubble size(s) in Mpc/h.
        d : int, float
            Separation in Mpc/h.

        """

        V_o = (4. * np.pi / 3.) * R**3 - np.pi * d * (R**2 - d**2 / 12.)

        if type(R) == np.ndarray:
            V_o[d >= 2 * R] = 0
        else:
            if d >= 2 * R:
                return 0.0

        return V_o

    def get_bb(self, z, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0., n_b=None,
        separate=False, **_kw_):
        """
        Comptute <bb'> following FZH04 model.
        """

        if Q == 0 or (not self.bubbles):
            return np.zeros_like(self.tab_R)

        pb = ProgressBar(self.tab_R.size, use=self.use_pbar,
            name="<bb'>(z={})".format(z))
        pb.start()

        P1 = []
        P2 = []
        for i, RR in enumerate(self.tab_R):
            pb.update(i)
            P1.append(self.get_P1(RR, Q=Q, R=R, sigma=sigma, alpha=alpha,
                gamma=gamma, n_b=n_b))
            P2.append(self.get_P2(RR, Q=Q, R=R, sigma=sigma, alpha=alpha,
                gamma=gamma, n_b=n_b))

        pb.finish()
        P1 = np.array(P1)
        P2 = np.array(P2)

        # Do we hit P1 with  (1. - Q)?
        if separate:
            return P1, P2
        else:
            return P1 + (1. - P1) * P2

    def get_bn(self, z, Q=0.0, R=5., sigma=0.5, gamma=0., alpha=0., n_b=None,
        separate=False, **_kw_):

        P1e = [self.get_P1(RR, Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha,
            n_b=n_b, exclusion=1) for RR in self.tab_R]
        P1e = np.array(P1e)

        #bb = self.get_bb(z, Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha,
        #    n_b=n_b)

        if separate:
            return P1e, (1. - Q)
        else:
            return P1e * (1. - Q)

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

    def get_bd(self, z, Q=0.0, R=5., sigma=0.5, alpha=0., n_b=None, bbar=1,
        bhbar=1, **_kw_):
        """
        Get the cross correlation function between bubbles and density, equivalent to <bd'>.
        """

        dd = self.get_dd(z)

        Rdiffabs=np.abs(self.Rarr - R)
        Rindex = Rdiffabs.argmin()


        fact =  bbar * bhbar * Q * dd

        xbar=Q
        #true formula is (1-xbar) = exp(-Q)
#        xbar= 1 - np.exp(-Q)


        bd = (1-xbar) * fact
        bd[self.Rarr <  R] = (1-xbar) *(1.0 - np.exp(fact[Rindex]) )
        #R is characteristic bubble size


        return bd

    def Rd(self, z, Q=0.0, R=5., sigma=0.5, gamma=0, alpha=0., n_b=None,
        bbar=1, bhbar=1):
        """
        Normalize the cross correlation

        """

        bd = self.get_bd(z, Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha,
            n_b=n_b, bbar=bbar, bhbar=bhbar)
        dd = self.get_dd(z)
        bb = self.get_bb(z, Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha,
            n_b=n_b)

        return (bd/np.sqrt(bb*dd))

    def get_variance_matter(self, z, R, kmin=1e-5, kmax=1e5, rtol=1e-5,
        atol=1e-5):
        """
        Return the variance in the matter field at redshift `z` when
        smoothing on scale `R`.
        """

        ikw = dict(epsrel=rtol, epsabs=atol, limit=10000, full_output=1)

        Pofk = lambda k: self.get_ps_matter(z, k)
        Wofk = lambda k: 3 * (np.sin(k * R) - k * R * np.cos(k * R)) \
            / (k * R)**3

        integrand_full = lambda k: Pofk(k) * np.abs(Wofk(k)**2) \
            * 4. * np.pi * k**2 / (2. * np.pi)**3

        kcrit = 1. / R
        norm = 1. / integrand_full(kmax)

        integrand_1 = lambda k: norm * Pofk(k) * 4. * np.pi * k**2 \
            * 3 / (k * R)**3 / (2. * np.pi)**3
        integrand_2 = lambda k: norm * Pofk(k) * 4. * np.pi * k**2 \
            * 3 * k * R / (k * R)**3 / (2. * np.pi)**3

        var = quad(integrand_full, kmin, kcrit, **ikw)[0]
        new = quad(integrand_1, kcrit, kmax, weight='sin', wvar=R, **ikw)[0] \
            - quad(integrand_2, kcrit, kmax, weight='cos', wvar=R, **ikw)[0]

        var += new / norm

        return var

    def get_density_threshold(self, z, Q=0.0, R=5., sigma=0.5,
        gamma=0, alpha=0, n_b=None, **_kw_):
        """
        Use "volume matching" to determine density level above which
        gas is ionized.

        Returns
        -------
        Both the density of bubbles and sigma_R of the density field
        smoothed on the appropriate scale.

        """

        # Hack!
        if (Q < tiny_Q) or (Q == 1):
            return -1, 0.0

        if self.use_volume_match == 1:
            bsd = self.get_bsd(Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha,
                n_b=n_b)
            # convert to dn/dlogR
            bsd = bsd * self.tab_R
            # weight by volume
            bsd = bsd * 4. * np.pi * self.tab_R**3 / 3.
            # Doesn't matter here but OK.
            bsd = bsd / Q
            # find peak in V dn/dlnR
            Rsm = self.tab_R[np.argmax(bsd)]
        elif self.use_volume_match == 2:
            bsd = self.get_bsd(Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha,
                n_b=n_b)
            # weight by volume
            bsd = bsd * 4. * np.pi * self.tab_R**3 / 3.
            # find peak in V dn/dR
            Rsm = self.tab_R[np.argmax(bsd)]
        elif self.use_volume_match == 3:
            Rsm = R
        elif self.use_volume_match == 10:
            bb1, bb2 = self.get_bb(z, Q=Q, R=R, sigma=sigma, gamma=gamma,
                alpha=alpha, n_b=n_b, separate=True)
            bb = bb1 + bb2
            P1_frac = bb1 / bb
            Rsm = np.interp(0.75, P1_frac[-1::-1], self.tab_R[-1::-1])
        elif self.use_volume_match == 11:
            bb1, bb2 = self.get_bb(z, Q=Q, R=R, sigma=sigma, gamma=gamma,
                alpha=alpha, n_b=n_b, separate=True)
            bb = bb1 + bb2
            P2_frac = bb2 / bb
            Rsm = np.interp(0.75, P2_frac, self.tab_R)
        else:
            raise NotImplemented('help')

        var_R = self.get_variance_matter(z, R=Rsm)
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
        n_b=None, **_kw_):
        """
        Return mean density in ionized regions.
        """

        # Hack!
        if (Q < tiny_Q) or (Q == 1):
            return 0.0

        x_thresh, w = self.get_density_threshold(z, Q=Q, R=R,
            sigma=sigma, gamma=gamma, alpha=alpha, n_b=n_b, **_kw_)

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
        gamma=0., alpha=0., n_b=None, beta=1., delta_ion=0., separate=False,
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

        bb = self.get_bb(z, Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha,
            n_b=n_b)
        bn = self.get_bn(z, Q=Q, R=R, sigma=sigma, gamma=gamma, alpha=alpha,
            n_b=n_b)
        dd = self.get_dd(z)
        _alpha = self.get_alpha(z, Ts)
        #beta_d, beta_T, beta_mu, beta_mu_T = self.get_betas(z, Ts)
        beta_phi, beta_mu = self.get_betas(z, Ts)
        beta_sq = (beta_mu**2 + beta_phi**2 + 2 * beta_mu * beta_phi)

        if not self.include_cross_terms:
            d_i = 0
        elif self.use_volume_match:
            d_i = self.get_bubble_density(z, Q=Q, R=R, sigma=sigma,
                gamma=gamma, alpha=alpha, n_b=n_b)
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
            bd_1pt = bbd = bbdd = np.zeros_like(self.tab_R)
            bdd = d_i * d_i * bb + d_i * d_n * bn
        else:
            raise NotImplemented('Only know include_cross_terms=1,2,3!')

        # RSDs
        # In all of these approaches, we're doing the average over \mu
        # here straight-away.
        #if self.include_rsd == 1:
        #    corr1 = (1. + beta_phi + self.get_rsd_int_mu2(self.include_mu_gt))
        #    corr2 = beta_sq
        #elif self.include_rsd == 2:
        #    mu_sq_avg = np.sqrt(self.get_rsd_boost_dd(-1) - 1)
        #    corr1 = (1. + beta_phi + mu_sq_avg)
        #    corr2 = (1. + beta_phi + mu_sq_avg)**2
        #elif self.include_rsd == 3:
        #    mu_sq_avg = 1.
        #    corr1 = (1. + beta_phi + mu_sq_avg)
        #    corr2 = (1. + beta_phi + mu_sq_avg)**2
        #else:
        #    corr1 = 1. + beta_phi
        #    corr2 = (1. + beta_phi)**2
#
        #bd *= corr1
        #bdd *= corr2
        #bbdd *= corr2

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
        alpha=0., n_b=None, delta_ion=0.):

        bb = 1 * self.get_bb(z, Q, R=R, sigma=sigma, alpha=alpha, gamma=gamma,
            n_b=n_b)
        dd = 1 * self.get_dd(z)
        _alpha = self.get_alpha(z, Ts)

        dTb = self.get_dTb_bulk(z, Ts=Ts)

        avg_term = _alpha**2 * Q**2

        # Include correlations in density and temperature caused by
        # adiabatic expansion/contraction.
        beta_phi, beta_mu = self.get_betas(z, Ts)

        dd *= (beta_mu + beta_phi)**2

        cf_21 = bb * _alpha**2 + dd - avg_term

        bd, bbd, bdd, bbdd, bbd_1pt, bd_1pt = \
            self.get_cross_terms(z, Q=Q, Ts=Ts, R=R, sigma=sigma, gamma=gamma,
                alpha=alpha, n_b=n_b, delta_ion=delta_ion, separate=True)

        bd *= (beta_mu + beta_phi)
        bdd *= (beta_mu + beta_phi)**2
        bbdd *= (beta_mu + beta_phi)**2

        cf_21 += bd + bbd + bdd + bbdd + bbd_1pt + bd_1pt

        return dTb**2 * cf_21

    def get_ps_21cm(self, z, k, Q=0.0, Ts=np.inf, R=5., sigma=0.5,
        gamma=0., alpha=0., n_b=None, delta_ion=0.):

        # Much faster without bubbles -- just scale P_mm
        if (not self.bubbles) or (Q < tiny_Q):
            ps_mm = np.array([self.get_ps_matter(z, kk) for kk in k])
            beta_phi, beta_mu = self.get_betas(z, Ts)
            beta_sq = (beta_mu + beta_phi)**2
            Tavg = self.get_dTb_avg(z, Q=Q, Ts=Ts, R=R, sigma=sigma,
                gamma=gamma, alpha=alpha, n_b=n_b)

            ps_21 = Tavg**2 * ps_mm * beta_sq

        else:
            # In this case, if include_rsd==True, each term will carry
            # its own correction term, so we don't apply a correction
            # explicitly here as we do above in the Q=0 density-driven limit.
            cf_21 = self.get_cf_21cm(z, Q=Q, Ts=Ts, R=R,
                sigma=sigma, gamma=gamma, alpha=alpha, n_b=n_b,
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

    def get_3d_realization(self, z, Lbox=100., vox=1., Q=0.0, Ts=np.inf,
        R=5., sigma=0.5, gamma=0., n_b=None, beta=1., use_kdtree=True,
        include_rho=True):
        """
        Make a 3-d realization representative of this model.

        .. note :: This just draws bubbles from the desired bubble size
            distribution and positions them randomly in a box.

        Parameters
        ----------
        z : int, float
            Redshift of interest.
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
        A tuple containing (box of ones and zeros, density box, 21-cm box),
        each of which are a 3-d array with dimensions [Lbox / vox]*3.

        """

        Npix = int(Lbox / vox)
        pdf = self.get_bsd(Q=Q, R=R, sigma=sigma, gamma=gamma, n_b=n_b)
        cdf = self.get_bsd_cdf(Q=Q, R=R, sigma=sigma, gamma=gamma, n_b=n_b)
        num_per_vol = self.get_nb(Q=Q, R=R, sigma=sigma, gamma=gamma, n_b=n_b)

        num = int(num_per_vol * Lbox**3)

        bins = np.arange(0, Lbox+vox, vox)
        binc = np.arange(0.5*vox, Lbox, vox)

        xx, yy, zz = np.meshgrid(binc, binc, binc)

        # Randomly generate `num` bubbles with sizes drawn from BSD.
        n = np.random.rand(num)
        R = np.exp(np.interp(np.log(n), np.log(cdf), np.log(self.tab_R)))

        # Randomly generate (x, y, z) positions for all bubbles
        p_len = np.random.rand(num*3).reshape(num, 3) * Lbox
        # Get bubble positions in terms of array indices
        p_bin = np.digitize(p_len, bins) - 1

        # Initialize a box. We'll zero-out elements lying within bubbles below.
        box = np.ones([binc.size]*3)

        # Can speed things up with a kdtree if you want.
        if use_kdtree:
            pos = np.array([xx.ravel(), yy.ravel(), zz.ravel()]).T
            kdtree = cKDTree(pos, boxsize=Lbox)

        # Loop over bubbles and flag all cells within them
        for h in range(p_bin.shape[0]):

            # Brute force
            if not use_kdtree:
                i, j, k = p_bin[h]
                dr = np.sqrt((xx - xx[i,j,k])**2 + (yy - yy[i,j,k])**2 \
                   + (zz - zz[i,j,k])**2)
                in_bubble = dr <= R[h]
                box[in_bubble] = 0
                continue

            # Spee-up with kdtree
            p = p_bin[h]

            # `neaRy` are indices in `pos`, i.e., not (i, j, k) indices
            d, neaRy = kdtree.query(p, k=1e4, distance_uppeRound=R * 10)

            in_bubble = d <= R[h]
            for elem in neaRy[in_bubble==True]:
                a, b, c = pos[elem]
                i, j, k = np.digitize([a, b, c], bins) - 1
                box[i,j,k] = 0

        # Set bulk IGM temperature
        dTb = box * self.get_dTb_bulk(z, Ts=Ts)

        if include_rho:
            power = lambda k: self.get_ps_matter(z=z, k=k)
            rho = pbox.LogNormalpowerbox(N=Npix, dim=3, pk=power,
                boxlength=Lbox).delta_x()
            dTb *= (1. + rho)
        else:
            rho = None

        return box, rho, dTb
