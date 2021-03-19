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

class BubbleModel(object):
    def __init__(self, bubbles=True, bubbles_ion=True,
        bubbles_pdf='lognormal', bubbles_Rfree=True,
        include_adiabatic_fluctuations=True, include_P1_corr=True,
        include_cross_terms=0, include_rsd=False, include_mu_gt=-1.,
        use_volume_match=1,
        Rmin=1e-2, Rmax=1e3, NR=1000, density_pdf='lognormal',
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
            'plexp'. The two parameters Rb and sigma_b characterize the PDF, and
            are the avg and rms of radii for 'lognormal'. For 'plexp' Rb is the
            critical radius, and sigma_b is a power-law index added to the usual 3.

        Array Setup
        -----------
        Rmin, Rmax : float
            Limits of configuration space integrals.
        NR : int
            Number of R bins to use.

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

        self.params = ['Ts']
        if self.bubbles:
            if self.bubbles_Rfree:
                self.params.extend(['Q', 'R_b', 'sigma_b'])
            else:
                self.params.extend(['Q', 'n_b', 'sigma_b'])

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
            nonlin = camb.model.NonLinear_both

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
        return self.cosmo.TCMB * (1. + z)

    def get_Tgas(self, z):
        return Tgadiabaticfit(z)

    def get_alpha(self, z, Ts):
        if not self.bubbles:
            return 0.
        elif self.bubbles_ion:
            return -1
        else:
            return self.get_Tcmb(z) / (Ts - self.get_Tcmb(z))

    def get_CT(self,z,Ts):
        # if self.bubbles:
        #     return 0.0 #we do not include it for the cases with bubbles, only for density (revisit)
        # else:
        return CTfit(z) * min(1.0,Ts/self.get_Tgas(z))

    def get_contrast(self, z, Ts):
        return 1. - self.get_Tcmb(z) / Ts

    #def get_phi(self, z, Ts):
    #    return self.get_Tcmb(z) / (Ts - self.get_Tcmb(z))

    #def get_betam(self,z,Ts):
    #    #bias, \delta T21 = betam \delta_m
    #    # if self.bubbles:
    #    #     return 1.0 #we do not include it for the cases with bubbles, only for density (revisit). Bias 1 in this case.
    #    # else:
    #    return 1.0 + self.get_CT(z,Ts) * self.get_Tcmb(z) / (Ts - self.get_Tcmb(z))

    def get_ps_matter(self, z, k):
        if not hasattr(self, '_matter_ps_'):
            self._init_cosmology()

        #self.cosmo.set_matter_power(redshifts=[z], kmax=k.max())
        #results = camb.get_results(self.cosmo)

        #kh, z, pk = results.get_matter_power_spectrum(minkh=k.min(), maxkh=k.max(), npoints=k.size)

        return self._matter_ps_.P(z, k)

    def get_Tcmb(self, z):
        return self.cosmo.TCMB * (1. + z)

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

    def _get_Rb_from_nb(self, Q=0.5, sigma_b=0.1, n_b=None,
        Qtol=1e-6, maxiter=10000):

        # Need to do this iteratively.

        if not hasattr(self, '_cache_Rb'):
            self._cache_Rb = {}

        # Cache seems to cause problems
        if (Q, sigma_b, n_b, Qtol) in self._cache_Rb.keys():
            #if Q > 0.01:
            #    print('found cached copy', (Q, sigma_b, n_b, Qtol),
            #        self._cache_bsd.keys())
            #input('<enter>')
            return self._cache_Rb[(Q, sigma_b, n_b, Qtol)]

        logRarr = np.log(self.tab_R)

        # Initial guess for bubble size
        logRb = np.log(2.)
        Rb = np.exp(logRb)
        logRhist = []

        logRstep = 0.1

        # Iterate until we achieve requested Q.
        ct = 0
        while ct < maxiter:
            logRhist.append(logRb)

            _bsd = self._get_bsd_unnormalized(Q=Q, R_b=np.exp(logRb),
                sigma_b=sigma_b, n_b=n_b)

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
                logRb -= logRstep
            else:
                logRb += logRstep

            ct += 1

        # This shouldn't ever happen but throw a warning if we hit `maxiter`.
        if ct == maxiter:
            print("WARNING: maxiter={} reached for Q={}, n_b={}, sigma_b={}".format(
                maxiter, Q, n_b, sigma_b
            ))
            print("(actual Q={})".format(_Q_))

        # Little kludge to ensure we get Q exactly in LS limit? Dep. on Qtol.
        #bsd *= Q / _Q_

        # Once we find the right bubble size, cache and return.
        self._cache_Rb[(Q, sigma_b, n_b, Qtol)] = np.exp(logRb)

        return self._cache_Rb[(Q, sigma_b, n_b, Qtol)]

    def _get_bsd_unnormalized(self, Q=0.5, R_b=5., sigma_b=0.1, n_b=None):
        logRarr = np.log(self.tab_R)
        logRb = np.log(R_b)
        if self.bubbles_pdf == 'lognormal':
            bsd = np.exp(-(logRarr - logRb)**2 / 2. / sigma_b**2)
        elif self.bubbles_pdf == 'plexp':
            bsd = (self.tab_R / R_b)**(3.0+sigma_b) \
                * np.exp(-self.tab_R / R_b)
        else:
            raise NotImplemented("Unrecognized `bubbles_pdf`: {}".format(
                self.bubbles_pdf))

        return bsd

    def get_bsd(self, Q=0.5, R_b=5., sigma_b=0.1, n_b=None):
        """
        Compute the bubble size distribution (BSD).

        Parameters
        ----------
        Q : int, float
            Fraction of volume filled by bubbles. Normalizes the BSD.
        R_b : int, float
            Typical bubble size [cMpc / h].
        sigma_b : int, float
            Another free parameter whose meaning depends on value of
            `bubbles_pdf` attribute set in constructor. For default `lognormal`
            BSD, this characterizes the width of the distribution. For `plexp`
            this is the power-law slope (minus 3).

        """

        if not self.bubbles_Rfree:
            R_b = self._get_Rb_from_nb(Q=Q, sigma_b=sigma_b, n_b=n_b)

        # Should cache bsd too.
        _bsd = self._get_bsd_unnormalized(Q=Q, R_b=R_b, sigma_b=sigma_b,
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

    def get_bsd_cdf(self, Q=0.5, R_b=5., sigma_b=0.1, n_b=None):
        """
        Compute the cumulative distribution function for the bubble size dist.
        """

        pdf = self.get_bsd(Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b)
        cdf = cumtrapz(pdf * self.tab_R, x=np.log(self.tab_R), initial=0.0)

        return cdf / cdf[-1]

    def get_nb(self, Q=0.5, R_b=5., sigma_b=0.1, n_b=None):
        """
        Compute the number density of bubbles [(h / Mpc)^3].
        """
        pdf = self.get_bsd(Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b)
        return np.trapz(pdf * self.tab_R, x=np.log(self.tab_R))

    def get_P1(self, d, Q=0.5, R_b=5., sigma_b=0.1, n_b=None, exclusion=0,
        use_corr=True):
        """
        Compute 1 bubble term.
        """

        bsd = self.get_bsd(Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b)
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
            P1e = self.get_P1(d, Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b,
                exclusion=1, use_corr=False)
            P1 *= (1. - P1e)

        return P1

    def get_P2(self, d, Q=0.5, R_b=5., sigma_b=0.1, n_b=None, xi_bb=0.):
        """
        Compute 2 bubble term.
        """

        if not self.approx_small_Q:
            return Q**2

        bsd = self.get_bsd(Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b)
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

    def get_bb(self, z, Q=0.5, R_b=5., sigma_b=0.1, n_b=None,
        separate=False, **_kw_):
        """
        Comptute <bb'> following bare-bones FZH04 model.
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
            P1.append(self.get_P1(RR, Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b))
            P2.append(self.get_P2(RR, Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b))

        pb.finish()
        P1 = np.array(P1)
        P2 = np.array(P2)

        # Do we hit P1 with  (1. - Q)?
        if separate:
            return P1, P2
        else:
            return P1 + (1. - P1) * P2

    def get_bn(self, z, Q=0.5, R_b=5., sigma_b=0.1, n_b=None,
        separate=False, **_kw_):

        P1e = [self.get_P1(RR, Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b,
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

        """

        if not hasattr(self, '_cache_dd_'):
            self._cache_dd_ = {}

        if z in self._cache_dd_:
            return self._cache_dd_[z]

        dd = get_cf_from_ps(self.tab_R, lambda kk: self.get_ps_matter(z, kk))

        self._cache_dd_[z] = dd
        return dd

    def get_bd(self, z, Q=0.5, R_b=5., sigma_b=0.1, n_b=None, bbar=1,
        bhbar=1, **_kw_):
        """
        Get the cross correlation function between bubbles and density, equivalent to <bd'>.

        """

        dd = self.get_dd(z)

        Rdiffabs=np.abs(self.Rarr - R_b)
        Rbindex = Rdiffabs.argmin()


        fact =  bbar * bhbar * Q * dd

        xbar=Q
        #true formula is (1-xbar) = exp(-Q)
#        xbar= 1 - np.exp(-Q)


        bd = (1-xbar) * fact
        bd[self.Rarr <  R_b] = (1-xbar) *(1.0 - np.exp(fact[Rbindex]) )
        #R_b is characteristic bubble size


        return bd

    def r_bd(self, z, Q=0.5, R_b=5., sigma_b=0.1, n_b=None, bbar=1, bhbar=1):
        """
        Normalize the cross correlation

        """

        bd = self.get_bd(z, Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b,
            bbar=bbar, bhbar=bhbar)
        dd = self.get_dd(z)
        bb = self.get_bb(z, Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b)

        return (bd/np.sqrt(bb*dd))

    def get_variance_matter(self, z, R):
        """
        Return the variance in the matter field at redshift `z` when smoothing
        on scale `R`.
        """
        Pofk = lambda k: self.get_ps_matter(z, k)
        Wofk = lambda k: 3 * (np.sin(k * R) - k * R * np.cos(k * R)) / (k * R)**3

        integrand = lambda k: Pofk(k) * np.abs(Wofk(k)**2) * 4. * np.pi * k**2 \
            / (2. * np.pi)**3

        var = quad(integrand, 0, np.inf, limit=100000)[0]

        return var

    def get_bubble_density(self, z, Q=0.5, R_b=5., sigma_b=0.1, n_b=None,
        **_kw_):
        """
        Return mean density in ionized regions.
        """

        if self.use_volume_match == 1:
            bsd = self.get_bsd(Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b)
            bsd *= 4. * np.pi * self.tab_R**3 / 3.
            # This is dn/dR. Does it matter that it's not dn/dlogR?
            bsd /= Q # Doesn't matter here
            Rsm = self.tab_R[np.argmax(bsd)]
        elif self.use_volume_match == 2:
            Rsm = R_b
        elif self.use_volume_match == 10:
            bb1, bb2 = self.get_bb(z, Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b,
                separate=True)
            bb = bb1 + bb2
            P1_frac = bb1 / bb
            Rsm = np.interp(0.75, P1_frac[-1::-1], self.tab_R[-1::-1])
        elif self.use_volume_match == 11:
            bb1, bb2 = self.get_bb(z, Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b,
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
        Pofx = lambda x: np.exp(-x**2 / 2. / w**2) \
            / np.sqrt(2 * np.pi) / w

        # Could do this analytically too.
        norm = quad(lambda x: Pofx(x), x_thresh, np.inf,
            limit=100000)[0]

        # Sanity check: do numerically
        #del_i = quad(lambda x: Pofx(x) * x, x_thresh, np.inf)[0]

        if self.density_pdf.lower() in ['normal', 'gaussian']:
            del_i = np.exp(-x_thresh**2 / 2. / w**2) * w \
                / np.sqrt(2 * np.pi)
        else:
            del_i = 0.5 * (-1. + np.exp(w**2 / 2.) \
                * (1. + erf((w**2 - x_thresh) / np.sqrt(2.) / w)) \
                + erf(x_thresh / np.sqrt(2.) / w))

        return del_i / norm

    def get_cross_terms(self, z, Q=0.5, Ts=np.inf, R_b=5., sigma_b=0.1,
        n_b=None, beta=1., delta_ion=0., separate=False, **_kw_):

        if not self.include_cross_terms:
            arr = np.zeros_like(self.tab_R)
            if separate:
                return [arr]* 6
            else:
                return arr

        bb = self.get_bb(z, Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b)
        bn = self.get_bn(z, Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b)
        dd = self.get_dd(z)
        alpha = self.get_alpha(z, Ts)

        if self.use_volume_match:
            d_i = self.get_bubble_density(z, Q=Q, R_b=R_b, sigma_b=sigma_b,
                n_b=n_b)
        else:
            d_i = delta_ion

        d_n = -d_i * Q / (1. - Q)

        # Currently neglects terms containing b and b' (other than <bb'>)
        if self.include_cross_terms == 1:
            bd = d_i * bb + d_n * bn
            bd_1pt = np.zeros_like(self.tab_R)
            bbd = np.zeros_like(self.tab_R)
            bbdd = Q**2 * dd#np.zeros_like(self.tab_R)
            bdd = Q * dd #d_i * d_i * bb + d_i * d_n * bn
        elif self.include_cross_terms == 2:
            bd = d_i * bb + d_n * bn
            bd_1pt = bbd = bbdd = np.zeros_like(self.tab_R)
            bdd = d_i * d_i * bb + d_i * d_n * bn
        elif self.include_cross_terms == 3:
            bd = d_i * bb + d_n * bn
            bd_1pt = d_i * Q

            bbd = d_i * bb
            bdd = d_i * d_i * bb + d_i * d_n * bn
            bbdd = bb * d_i**2
        else:
            raise NotImplemented('Only know include_cross_terms=1!')

        if separate:
            return 2 * alpha * bd, 2 * alpha**2 * bbd, 2 * alpha * bdd, \
                alpha**2 * bbdd, \
                - 2 * alpha**2 * Q * bd_1pt, -alpha**2 * bd_1pt**2
        else:
            return 2 * alpha * bd + 2 * alpha**2 * bbd + 2 * alpha * bdd \
                + alpha**2 * bbdd \
                - 2 * alpha**2 * Q * bd_1pt - alpha**2 * bd_1pt**2

    def get_cf_21cm(self, z, Q=0.5, Ts=np.inf, R_b=5., sigma_b=0.1,
        n_b=None, delta_ion=0.):

        bb = self.get_bb(z, Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b)
        dd = self.get_dd(z)
        alpha = self.get_alpha(z, Ts)
        con = self.get_contrast(z, Ts)
        CT = self.get_CT(z, Ts)

        dTb = self.get_dTb_bulk(z, Ts=Ts)

        avg_term = alpha**2 * Q**2

        cf_21 = bb * alpha**2 + dd - avg_term

        # Include correlations in density and temperature caused by
        # adiabatic expansion/contraction.
        if self.include_adiabatic_fluctuations:
            cf_21 += dd * (2 * CT / con + (CT / con)**2)

        if self.include_cross_terms:
            new = self.get_cross_terms(z, Q=Q, R_b=R_b, sigma_b=sigma_b,
                n_b=n_b, delta_ion=delta_ion, separate=False)
            cf_21 += new

        return dTb**2 * cf_21

    def get_ps_21cm(self, z, k, Q=0.5, Ts=np.inf, R_b=5., sigma_b=0.1,
        n_b=None, delta_ion=0.):

        # Much faster without bubbles -- just scale P_mm
        if (not self.bubbles) or (Q == 0.):
            ps_mm = np.array([self.get_ps_matter(z, kk) for kk in k])
            if self.include_adiabatic_fluctuations:
                con = self.get_contrast(z, Ts)
                CT = self.get_CT(z, Ts)
                betam = 1. + (2 * CT / con + (CT / con)**2)
            else:
                betam = 1.

            ps_21 = self.get_dTb_bulk(z, Ts=Ts)**2 * ps_mm * betam
        else:
            cf_21 = self.get_cf_21cm(z, Q=Q, Ts=Ts, R_b=R_b, sigma_b=sigma_b,
                n_b=n_b, delta_ion=delta_ion)

            # Setup interpolant
            _fcf = interp1d(np.log(self.tab_R), cf_21, kind='cubic',
                bounds_error=False, fill_value=0.)
            f_cf = lambda RR: _fcf.__call__(np.log(RR))

            if type(k) != np.ndarray:
                k = np.array([k])

            ps_21 = get_ps_from_cf(k, f_cf=f_cf, Rmin=self.tab_R.min(),
                Rmax=self.tab_R.max())

        #
        if self.include_rsd:
            ps_21 *= self.get_rsd_boost(self.include_mu_gt)

        return ps_21

    def get_rsd_boost(self, mu):
        # This is just \int_{\mu_{\min}}^1 (1 + \mu^2)^2
        mod = (1. - mu) + 2. * (1. - mu**3) / 3. + 0.2 * (1. - mu**5)
        return mod / (1. - mu)

    def get_3d_realization(self, z, Lbox=100., vox=1., Q=0.5, Ts=np.inf, R_b=5.,
        sigma_b=0.1, n_b=None, beta=1., use_kdtree=True, include_rho=True):
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
        pdf = self.get_bsd(Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b)
        cdf = self.get_bsd_cdf(Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b)
        num_per_vol = self.get_nb(Q=Q, R_b=R_b, sigma_b=sigma_b, n_b=n_b)

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

            # `nearby` are indices in `pos`, i.e., not (i, j, k) indices
            d, nearby = kdtree.query(p, k=1e4, distance_upper_bound=R_b * 10)

            in_bubble = d <= R[h]
            for elem in nearby[in_bubble==True]:
                a, b, c = pos[elem]
                i, j, k = np.digitize([a, b, c], bins) - 1
                box[i,j,k] = 0

        # Set bulk IGM temperature
        dTb = box * self.get_dTb_bulk(z, Ts=Ts)

        if include_rho:
            power = lambda k: self.get_ps_matter(z=z, k=k)
            rho = pbox.LogNormalPowerBox(N=Npix, dim=3, pk=power,
                boxlength=Lbox).delta_x()
            dTb *= (1. + rho)
        else:
            rho = None

        return box, rho, dTb
