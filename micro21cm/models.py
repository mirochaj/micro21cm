"""

models.py

Authors: Jordan Mirocha and Julian B. Munoz
Affiliation: McGill University and Harvard-Smithsonian Center for Astrophysics
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import camb
import numpy as np
import powerbox as pbox
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, quad
from scipy.special import erfcinv, erf, erfc
from .util import get_cf_from_ps, get_ps_from_cf, ProgressBar, CTfit, \
    Tgadiabaticfit

tiny_Q = 1e-3

class BubbleModel(object):
    def __init__(self, bubbles=True, bubbles_ion=True,
        bubbles_pdf='lognormal', bubbles_Rfree=True,
        include_adiabatic_fluctuations=True, include_P1_corr=True,
        include_cross_terms=1, include_rsd=2, include_mu_gt=-1.,
        use_volume_match=1, density_pdf='lognormal',
        Rmin=1e-2, Rmax=1e3, NR=1000,
        omega_b=0.0486, little_h=0.67, omega_m=0.3089, ns=0.96,
        transfer_kmax=500., transfer_k_per_logint=11, zmax=20.,
        use_pbar=False, approx_small_Q=False, approx_linear=True):
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
            is the critical radius, and sigma is a power-law index added to
            the usual 3.
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
        self.NR = NR
        self.bubbles = bubbles
        self.bubbles_ion = bubbles_ion
        self.bubbles_pdf = bubbles_pdf
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
                self.params.extend(['R', 'sigma'])
            else:
                self.params.extend(['n_b', 'sigma'])

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
            nonlin = camb.model.NonLineaRoth

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

    def get_dTb_avg(self, z, Q=0.0, R=5., sigma=0.1, n_b=None,
        Ts=np.inf):
        """
        Return volume-averaged 21-cm brightness temperature, i.e., the
        global 21-cm signal.

        .. note :: This is different from `get_dTb_bulk` (see next function)
            because appropriately weights by volume, and accounts for
            cross-correlations between ionization and density.

        """
        bd = self.get_bubble_density(z, Q=Q, R=R, sigma=sigma,
            n_b=n_b) * Q
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

    def _get_R_from_nb(self, Q=0.0, sigma=0.1, n_b=None,
        Qtol=1e-6, maxiter=10000, **_kw_):
        """
        If self.bubbles_Rfree == False, it means the bubble abundance, n_b,
        is our free parameter. In this case, we must iteratively solve for
        the characteristic bubble size needed to guarantee that our BSD
        integrates to Q.
        """
        # Need to do this iteratively.

        if not hasattr(self, '_cache_R'):
            self._cache_R = {}

        # Cache seems to cause problems
        if (Q, sigma, n_b, Qtol) in self._cache_R.keys():
            #if Q > 0.01:
            #    print('found cached copy', (Q, sigma, n_b, Qtol),
            #        self._cache_bsd.keys())
            #input('<enter>')
            return self._cache_R[(Q, sigma, n_b, Qtol)]

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
                sigma=sigma, n_b=n_b)

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
        self._cache_R[(Q, sigma, n_b, Qtol)] = np.exp(logR)

        return self._cache_R[(Q, sigma, n_b, Qtol)]

    def _get_bsd_unnormalized(self, Q=0.0, R=5., sigma=0.1, n_b=None):
        """
        Return an unnormalized version of the bubble size distribution.
        """
        logRarr = np.log(self.tab_R)
        logR = np.log(R)
        if self.bubbles_pdf == 'lognormal':
            bsd = np.exp(-(logRarr - logR)**2 / 2. / sigma**2)
        elif self.bubbles_pdf == 'plexp':
            bsd = (self.tab_R / R)**(3.0+sigma) \
                * np.exp(-self.tab_R / R)
        else:
            raise NotImplemented("Unrecognized `bubbles_pdf`: {}".format(
                self.bubbles_pdf))

        return bsd

    def _cache_bsd(self, Q=0.0, R=5., sigma=0.1, n_b=None):
        if not hasattr(self, '_cache_bsd_'):
            self._cache_bsd_ = {}

        key = (Q, R, sigma, n_b)
        if key in self._cache_bsd_:
            return self._cache_bsd_[key]

        return None

    def get_bsd(self, Q=0.0, R=5., sigma=0.1, n_b=None, **_kw_):
        """
        Compute the bubble size distribution (BSD).

        This is normalized so that:

        if self.approx_small_Q:
            \int bsd(R) V(R) dR = Q
        else:
            1 - \exp{-\int bsd(R) V(R) dR} = Q


        Parameters
        ----------
        Q : int, float
            Fraction of volume filled by bubbles. Normalizes the BSD.
        R : int, float
            Typical bubble size [cMpc / h].
        sigma : int, float
            Another free parameter whose meaning depends on value of
            `bubbles_pdf` attribute set in constructor. For default
            `lognormal` BSD, this characterizes the width of the
            distribution. For `plexp` this is the power-law slope (minus 3).

        """

        cached_bsd = self._cache_bsd(Q, R, sigma, n_b)
        if cached_bsd is not None:
            return cached_bsd

        if not self.bubbles_Rfree:
            R = self._get_R_from_nb(Q=Q, sigma=sigma, n_b=n_b)

        # Should cache bsd too.
        _bsd = self._get_bsd_unnormalized(Q=Q, R=R, sigma=sigma,
            n_b=n_b)

        integ = _bsd * 4 * np.pi * self.tab_R**3 / 3.
        norm = 1. / integ.max()
        integ = np.trapz(integ * self.tab_R * norm,
            x=np.log(self.tab_R)) / norm

        if self.approx_small_Q:
            corr = Q / integ
        else:
            # In this case, normalize n_b such that
            # 1 - \exp{\int n_b V_b dR} = Q
            corr = -1 * np.log(1 - Q) / integ

        # Normalize to provided ionized fraction
        bsd = _bsd * corr

        return bsd

    def get_bsd_cdf(self, Q=0.0, R=5., sigma=0.1, n_b=None):
        """
        Compute the cumulative distribution function for the bubble size dist.
        """

        pdf = self.get_bsd(Q=Q, R=R, sigma=sigma, n_b=n_b)
        cdf = cumtrapz(pdf * self.tab_R, x=np.log(self.tab_R), initial=0.0)

        return cdf / cdf[-1]

    def get_nb(self, Q=0.0, R=5., sigma=0.1, n_b=None):
        """
        Compute the number density of bubbles [(h / Mpc)^3].
        """
        pdf = self.get_bsd(Q=Q, R=R, sigma=sigma, n_b=n_b)
        return np.trapz(pdf * self.tab_R, x=np.log(self.tab_R))

    def get_P1(self, d, Q=0.0, R=5., sigma=0.1, n_b=None, exclusion=0,
        use_corr=True):
        """
        Compute 1 bubble term.
        """

        bsd = self.get_bsd(Q, R=R, sigma=sigma, n_b=n_b)
        V_o = self.get_overlap_vol(self.tab_R, d)

        if exclusion:
            V = 4. * np.pi * self.tab_R**3 / 3.
            integ = np.trapz(bsd * (V - V_o) * self.tab_R,
                x=np.log(self.tab_R))
        else:
            integ = np.trapz(bsd * V_o * self.tab_R, x=np.log(self.tab_R))

        if self.approx_small_Q:
            P1 = integ
        else:
            P1 = 1. - np.exp(-integ)

        if self.include_P1_corr and use_corr and (not exclusion):
            P1e = self.get_P1(d, Q=Q, R=R, sigma=sigma, n_b=n_b,
                exclusion=1, use_corr=False)
            P1 *= (1. - P1e)

        return P1

    def get_P2(self, d, Q=0.0, R=5., sigma=0.1, n_b=None, xi_bb=0.):
        """
        Compute 2 bubble term.
        """

        if not self.approx_small_Q:
            return Q**2

        bsd = self.get_bsd(Q, R=R, sigma=sigma, n_b=n_b)
        V_o = self.get_overlap_vol(self.tab_R, d)

        V = 4. * np.pi * self.tab_R**3 / 3.

        integ1 = np.trapz(bsd * (V - V_o) * self.tab_R,
            x=np.log(self.tab_R))
        integ2 = np.trapz(bsd * (V - V_o) * (1. + xi_bb) * self.tab_R,
            x=np.log(self.tab_R))

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

    def get_bb(self, z, Q=0.0, R=5., sigma=0.1, n_b=None,
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
            P1.append(self.get_P1(RR, Q=Q, R=R, sigma=sigma, n_b=n_b))
            P2.append(self.get_P2(RR, Q=Q, R=R, sigma=sigma, n_b=n_b))

        pb.finish()
        P1 = np.array(P1)
        P2 = np.array(P2)

        # Do we hit P1 with  (1. - Q)?
        if separate:
            return P1, P2
        else:
            return P1 + (1. - P1) * P2

    def get_bn(self, z, Q=0.0, R=5., sigma=0.1, n_b=None,
        separate=False, **_kw_):

        P1e = [self.get_P1(RR, Q=Q, R=R, sigma=sigma, n_b=n_b,
            exclusion=1) for RR in self.tab_R]
        P1e = np.array(P1e)

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

    def get_bd(self, z, Q=0.0, R=5., sigma=0.1, n_b=None, bbar=1,
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

    def Rd(self, z, Q=0.0, R=5., sigma=0.1, n_b=None, bbar=1, bhbar=1):
        """
        Normalize the cross correlation

        """

        bd = self.get_bd(z, Q, R=R, sigma=sigma, n_b=n_b,
            bbar=bbar, bhbar=bhbar)
        dd = self.get_dd(z)
        bb = self.get_bb(z, Q, R=R, sigma=sigma, n_b=n_b)

        return (bd/np.sqrt(bb*dd))

    def get_variance_matter(self, z, R):
        """
        Return the variance in the matter field at redshift `z` when
        smoothing on scale `R`.
        """

        Pofk = lambda k: self.get_ps_matter(z, k)
        Wofk = lambda k: 3 * (np.sin(k * R) - k * R * np.cos(k * R)) \
            / (k * R)**3

        integrand = lambda k: Pofk(k) * np.abs(Wofk(k)**2) \
            * 4. * np.pi * k**2 / (2. * np.pi)**3

        var = quad(integrand, 0, np.inf, limit=100000)[0]

        return var

    def get_density_threshold(self, z, Q=0.0, R=5., sigma=0.1, n_b=None,
        **_kw_):
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
            bsd = self.get_bsd(Q=Q, R=R, sigma=sigma, n_b=n_b)
            bsd *= 4. * np.pi * self.tab_R**3 / 3.
            # This is dn/dR. Does it matter that it's not dn/dlogR?
            bsd /= Q # Doesn't matter here
            Rsm = self.tab_R[np.argmax(bsd)]
        elif self.use_volume_match == 2:
            Rsm = R
        elif self.use_volume_match == 10:
            bb1, bb2 = self.get_bb(z, Q=Q, R=R, sigma=sigma, n_b=n_b,
                separate=True)
            bb = bb1 + bb2
            P1_frac = bb1 / bb
            Rsm = np.interp(0.75, P1_frac[-1::-1], self.tab_R[-1::-1])
        elif self.use_volume_match == 11:
            bb1, bb2 = self.get_bb(z, Q=Q, R=R, sigma=sigma, n_b=n_b,
                separate=True)
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

    def get_bubble_density(self, z, Q=0.0, R=5., sigma=0.1, n_b=None,
        **_kw_):
        """
        Return mean density in ionized regions.
        """

        # Hack!
        if (Q < tiny_Q) or (Q == 1):
            return 0.0

        x_thresh, w = self.get_density_threshold(z, Q=Q, R=R,
            sigma=sigma, n_b=n_b, **_kw_)

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

    def get_cross_terms(self, z, Q=0.0, Ts=np.inf, R=5., sigma=0.1,
        n_b=None, beta=1., delta_ion=0., separate=False, **_kw_):
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

        bb = self.get_bb(z, Q=Q, R=R, sigma=sigma, n_b=n_b)
        bn = self.get_bn(z, Q=Q, R=R, sigma=sigma, n_b=n_b)
        dd = self.get_dd(z)
        alpha = self.get_alpha(z, Ts)
        #beta_d, beta_T, beta_mu, beta_mu_T = self.get_betas(z, Ts)
        beta_phi, beta_mu = self.get_betas(z, Ts)
        beta_sq = (beta_mu**2 + beta_phi**2 + 2 * beta_mu * beta_phi)

        if not self.include_cross_terms:
            d_i = 0
        elif self.use_volume_match:
            d_i = self.get_bubble_density(z, Q=Q, R=R, sigma=sigma,
                n_b=n_b)
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
            return 2 * alpha * bd, 2 * alpha**2 * bbd, 2 * alpha * bdd, \
                alpha**2 * bbdd, \
                - 2 * alpha**2 * Q * bd_1pt, -alpha**2 * bd_1pt**2
        else:
            return 2 * alpha * bd + 2 * alpha**2 * bbd + 2 * alpha * bdd \
                + alpha**2 * bbdd \
                - 2 * alpha**2 * Q * bd_1pt - alpha**2 * bd_1pt**2

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

    def get_cf_21cm(self, z, Q=0.0, Ts=np.inf, R=5., sigma=0.1,
        n_b=None, delta_ion=0.):

        bb = 1 * self.get_bb(z, Q, R=R, sigma=sigma, n_b=n_b)
        dd = 1 * self.get_dd(z)
        alpha = self.get_alpha(z, Ts)

        dTb = self.get_dTb_bulk(z, Ts=Ts)

        avg_term = alpha**2 * Q**2

        # Include correlations in density and temperature caused by
        # adiabatic expansion/contraction.
        beta_phi, beta_mu = self.get_betas(z, Ts)

        dd *= (beta_mu + beta_phi)**2

        cf_21 = bb * alpha**2 + dd - avg_term

        bd, bbd, bdd, bbdd, bbd_1pt, bd_1pt = \
            self.get_cross_terms(z, Q=Q, Ts=Ts, R=R, sigma=sigma,
            n_b=n_b, delta_ion=delta_ion, separate=True)

        bd *= (beta_mu + beta_phi)
        bdd *= (beta_mu + beta_phi)**2
        bbdd *= (beta_mu + beta_phi)**2

        cf_21 += bd + bbd + bdd + bbdd + bbd_1pt + bd_1pt

        return dTb**2 * cf_21

    def get_ps_21cm(self, z, k, Q=0.0, Ts=np.inf, R=5., sigma=0.1,
        n_b=None, delta_ion=0.):

        # Much faster without bubbles -- just scale P_mm
        if (not self.bubbles) or (Q < tiny_Q):
            ps_mm = np.array([self.get_ps_matter(z, kk) for kk in k])
            beta_phi, beta_mu = self.get_betas(z, Ts)
            beta_sq = (beta_mu + beta_phi)**2
            Tavg = self.get_dTb_avg(z, Q=Q, Ts=Ts, R=R, sigma=sigma,
                n_b=n_b)

            ps_21 = Tavg**2 * ps_mm * beta_sq

        else:
            # In this case, if include_rsd==True, each term will carry
            # its own correction term, so we don't apply a correction
            # explicitly here as we do above in the Q=0 density-driven limit.
            cf_21 = self.get_cf_21cm(z, Q=Q, Ts=Ts, R=R, sigma=sigma,
                n_b=n_b, delta_ion=delta_ion)

            # Setup interpolant
            _fcf = interp1d(np.log(self.tab_R), cf_21, kind='cubic',
                bounds_error=False, fill_value=0.)
            f_cf = lambda RR: _fcf.__call__(np.log(RR))

            if type(k) != np.ndarray:
                k = np.array([k])

            ps_21 = get_ps_from_cf(k, f_cf=f_cf, Rmin=self.tab_R.min(),
                Rmax=self.tab_R.max())

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
        R=5., sigma=0.1, n_b=None, beta=1., use_kdtree=True,
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
        pdf = self.get_bsd(Q=Q, R=R, sigma=sigma, n_b=n_b)
        cdf = self.get_bsd_cdf(Q=Q, R=R, sigma=sigma, n_b=n_b)
        num_per_vol = self.get_nb(Q=Q, R=R, sigma=sigma, n_b=n_b)

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
