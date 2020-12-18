"""

models.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import camb
import numpy as np
from scipy.interpolate import interp1d
from .util import get_cf_from_ps, get_ps_from_cf, ProgressBar

class BubbleModel(object):
    def __init__(self, bubbles=True, bubbles_ion=True, bubbles_pdf='lognormal',
        Rmin=1e-2, Rmax=1e4, NR=1000, cross_terms=0,
        omega_b=0.0486, little_h=0.67, omega_m=0.3089, ns=0.96,
        transfer_kmax=500., transfer_k_per_logint=11, zmax=20.,
        use_pbar=False, approx_small_Q=True):
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
            'plexp'.

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
        self.use_pbar = use_pbar
        self.approx_small_Q = approx_small_Q

        self.params = ['Ts']
        if self.bubbles:
            self.params.extend(['Q', 'R_b', 'sigma_b'])

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
             'ns': ns,
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

        # `P` method of `matter_ps` is function of (z, k)
        self._matter_ps_ = camb.get_matter_power_interpolator(self._cosmo_,
            **self.transfer_params)

    @property
    def cosmo(self):
        if not hasattr(self, '_cosmo_'):
            self._init_cosmology()
        return self._cosmo_

    def get_alpha(self, z, Ts):
        if self.bubbles_ion:
            return -1
        else:
            return self.get_Tcmb(z) / (Ts - self.get_Tcmb(z))

    def get_ps_matter(self, z, k):
        if not hasattr(self, '_matter_ps_'):
            self._init_cosmology()
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

    def get_bsd(self, Q=0.5, R_b=5., sigma_b=0.1):

        if self.bubbles_pdf == 'lognormal':
            logRarr = np.log(self.tab_R)
            logRb = np.log(R_b)
            n_b = np.exp(-(logRarr - logRb)**2 / 2. / sigma_b**2)
        elif self.bubbles_pdf == 'plexp':
            n_b = (self.tab_R / R_b)**sigma_b * np.exp(-self.tab_R / R_b)
        else:
            raise NotImplemented("Unrecognized `bubbles_pdf`: {}".format(
                self.bubbles_pdf))

        integ = n_b * 4 * np.pi * self.tab_R**2
        norm = 1. / integ.max()

        _Q_ = np.trapz(integ * self.tab_R * norm, x=np.log(self.tab_R)) / norm

        # Normalize to provided ionized fraction.
        n_b *= Q / _Q_

        return n_b

    def get_P1(self, d, Q=0.5, R_b=5., sigma_b=0.1):
        """
        Compute 1 bubble term.
        """

        n_b = self.get_bsd(Q, R_b=R_b, sigma_b=sigma_b)
        V_o = self.get_overlap_vol(self.tab_R, d)

        integ = np.trapz(n_b * V_o * self.tab_R, x=np.log(self.tab_R))

        if self.approx_small_Q:
            return integ
        else:
            return 1. - np.exp(-integ)

    def get_P2(self, d, Q=0.5, R_b=5., sigma_b=0.1, xi_bb=0.):
        """
        Compute 2 bubble term.
        """

        return Q**2

        # Subtlety here with using Q as a free parameter.
        # Re-visit this!
        n_b = self.get_bsd(Q, R_b=R_b, sigma_b=sigma_b)
        V_o = self.get_overlap_vol(self.tab_R, d)

        V = 4. * np.pi * self.tab_R**3 / 3.
        norm = 4 * np.pi * self.tab_R**2

        n_b *= norm / V

        # n_b normalized such that integral weighted by 4 * np.pi * self.tab_R**2
        # integrates to Q.

        integ1 = np.trapz(n_b * (V - V_o) * self.tab_R, x=np.log(self.tab_R))
        integ2 =np.trapz(n_b * (V - V_o) * (1. + xi_bb) * self.tab_R,
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

    def get_bb(self, z, Q=0.5, R_b=5., sigma_b=0.1):
        """
        Comptute <bb'> following bare-bones FZH04 model.
        """

        pb = ProgressBar(self.tab_R.size, use=self.use_pbar,
            name="<bb'>(z={})".format(z))
        pb.start()

        P1 = []
        P2 = []
        for i, RR in enumerate(self.tab_R):
            pb.update(i)
            P1.append(self.get_P1(RR, Q=Q, R_b=R_b, sigma_b=sigma_b))
            P2.append(self.get_P2(RR, Q=Q, R_b=R_b, sigma_b=sigma_b))

        pb.finish()
        P1 = np.array(P1)
        P2 = np.array(P2)

        return P1 * (1. - Q) + (1. - P1) * P2

    def get_dd(self, z):
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

    def get_bd(self, z, Q=0.5, R_b=5., sigma_b=0.1, bbar=1, bhbar=1):
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

    def r_bd(self, z, Q=0.5, R_b=5., sigma_b=0.1, bbar=1, bhbar=1):
        """
        Normalize the cross correlation

        """

        bd = self.get_bd(z, Q, R_b=R_b, sigma_b=sigma_b, bbar=bbar, bhbar=bhbar)
        dd = self.get_dd(z)
        bb = self.get_bb(z, Q, R_b=R_b, sigma_b=sigma_b)

        return (bd/np.sqrt(bb*dd))

    def get_cf_21cm(self, z, Q=0.5, Ts=np.inf, R_b=5., sigma_b=0.1, beta=1.):

        bb = self.get_bb(z, Q, R_b=R_b, sigma_b=sigma_b)
        dd = self.get_dd(z)
        alpha = self.get_alpha(z, Ts)

        dTb = self.get_dTb_bulk(z, Ts=Ts)

        if self.approx_small_Q:
            _Q_ = Q
        else:
            _Q_ = Q#(1. - np.exp(-Q))**2

        avg_term = alpha**2 * _Q_**2

        return dTb**2 * (bb * alpha**2 + dd * beta**2 - avg_term)

    def get_ps_21cm(self, z, k, Q=0.5, Ts=np.inf, R_b=5., sigma_b=0.1, beta=1.):

        # Much faster without bubbles -- just scale P_mm
        if not self.bubbles:
            ps_mm = np.array([self.get_ps_matter(z, kk) for kk in k])
            ps_21 = self.get_dTb_bulk(z, Ts=Ts)**2 * ps_mm
        else:
            cf_21 = self.get_cf_21cm(z, Q=Q, Ts=Ts, R_b=R_b, sigma_b=sigma_b,
                beta=beta)

            # Setup interpolant
            _fcf = interp1d(np.log(self.tab_R), cf_21, kind='cubic',
                bounds_error=False, fill_value=0.)
            f_cf = lambda RR: _fcf.__call__(np.log(RR))

            if type(k) != np.ndarray:
                k = np.array([k])

            ps_21 = get_ps_from_cf(k, f_cf=f_cf, Rmin=self.tab_R.min(),
                Rmax=self.tab_R.max())

        return ps_21
