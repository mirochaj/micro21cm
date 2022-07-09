"""

inference.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import os
import sys
import time
import pickle
import numpy as np
from scipy.special import erf
from .models import BubbleModel
from scipy.integrate import quad

try:
    import emcee
except ImportError:
    pass

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    from schwimmbad import MPIPool
except ImportError:
    rank = 0
    size = 1

m_H = 1.673532518188e-24
G = 6.673e-8     				# Gravitational constant - [G] = cm^3/g/s^2
km_per_mpc = 3.08568e19
c = 29979245800.0 				# Speed of light - [c] = cm/s
sigma_T = 6.65e-25			    # Thomson cross section [sigma_T] = cm^2
max_Q = 0.999999

_priors_broad = \
{
 'Ts': (1e-2, 1000.),
 'Q': (0, 1),
 'R': (0.5, 50),
 'sigma': (0.25, 2.5),
 'gamma': (-4, 0),
 'Asys': (0.5, 1.5),
}

_guesses_broad = \
{
 'Ts': (10., 1000.),
 'Q': (0.2, 0.8),
 'R': (0.5, 10.),
 'sigma': (0.6, 1.2),
 'gamma': (-3.8, -3),
 'Asys': (0.8, 1.2),
}

_bins = \
{
 'Ts': np.arange(0, 500, 10),
 'Q': np.arange(-0.01, 1.01, 0.01),
 'R': np.arange(0, 50, 0.2),
 'sigma': np.arange(0.0, 1.0, 0.05),
 'Asys': np.arange(0.5, 1.55, 0.05),
}

_guesses_Q_tanh = {'p0': (6, 10), 'p1': (1, 4)}
_guesses_Q_bpl = {'p0': (0.25, 0.75), 'p1': (6, 10), 'p2': (-3, -1),
    'p3': (-4, -2)}
_guesses_Q_pl = {'p0': (0.25, 0.75), 'p1': (-8, 0)}

_guesses_Q = {'tanh': _guesses_Q_tanh, 'bpl': _guesses_Q_bpl,
    'broad': _guesses_broad['Q'], 'pl': _guesses_Q_pl}

_guesses_R_pl = {'p0': (5, 15.), 'p1': (0, 2)}
_guesses_R = {'pl': _guesses_R_pl, 'broad': _guesses_broad['R']}

_guesses_T_dpl = {'p0': (5., 20.), 'p1': (8, 20), 'p2': (3, 7),
    'p3': (-2.5, -1.5)}
_guesses_T_pl = {'p0': (1., 30.), 'p1': (-10, 3.)}
_guesses_T = {'broad': _guesses_broad['Ts'], 'dpl': _guesses_T_dpl,
     'pl': _guesses_T_pl}

_guesses_s_lin = {'p0': (0.5, 1.5), 'p1': (-0.5, 0.5)}
_guesses_s_pl = {'p0': (0.3, 1.5), 'p1': (-0.5, 0.5)}
_guesses_s = {'pl': _guesses_s_pl, 'linear': _guesses_s_lin,
    'broad': _guesses_broad['sigma']}

_guesses_g = {'broad': _guesses_broad['gamma']}

_guesses_A_erf = {'p0': (1, 1.5), 'p1': (1, 3), 'p2': (0.6, 1)}
_guesses_A = {'erf': _guesses_A_erf, 'broad': _guesses_broad['Asys']}

_guesses = {'R': _guesses_R, 'Q': _guesses_Q, 'Ts': _guesses_T,
    'sigma': _guesses_s, 'gamma': _guesses_g, 'Asys': _guesses_A}

_priors_Q_tanh = {'p0': (5, 15), 'p1': (0, 20)}
_priors_Q_bpl = {'p0': (0, 1), 'p1': (5, 20), 'p2': (-6, 0), 'p3': (-6, 0)}
_priors_Q_pl = {'p0': (0, 1), 'p1': (-20, 0)}
_priors_Q = {'tanh': _priors_Q_tanh, 'bpl': _priors_Q_bpl,
    'broad': _priors_broad['Q'], 'pl': _priors_Q_pl}

_priors_R_pl = {'p0': (0.5, 50), 'p1': (0, 10)}
_priors_R = {'pl': _priors_R_pl, 'broad': _priors_broad['R']}

_priors_s_pl = {'p0': (0.0, 1), 'p1': (-2, 2)}
_priors_s_lin = {'p0': (0.0, 2), 'p1': (-2, 2)}

_priors_s = {'pl': _priors_s_pl, 'linear': _priors_s_lin,
    'broad': _priors_broad['sigma']}

_priors_T_dpl = {'p0': (0, 50), 'p1': (5, 30), 'p2': (0, 8), 'p3': (-6, 0)}
_priors_T_pl = {'p0': (0.1, 1e3), 'p1': (-20, 5)}
_priors_T = {'broad': _priors_broad['Ts'], 'dpl': _priors_T_dpl,
    'pl': _priors_T_pl}
_priors_g = {'broad': _priors_broad['gamma']}

_priors_A_erf = {'p0': (0.5, 1.5), 'p1': (0, 10), 'p2': (0.5, 1.5)}
_priors_A = {'broad': _priors_broad['Asys'], 'erf': _priors_A_erf}

_priors = {'Q': _priors_Q, 'R': _priors_R, 'sigma': _priors_s,
    'Ts': _priors_T, 'gamma': _priors_g, 'Asys': _priors_A}

fit_kwargs = \
{
 'prior_tau': False,
 'prior_GP': False,

 'Q_func': None,
 'R_func': None,
 'Ts_func': None,
 'sigma_func': None,
 'gamma_func': None,
 'Asys_func': None,

 'Q_val': None,
 'Ts_val': None,
 'R_val': None,
 'sigma_val': None,
 'gamma_val': None,
 'Asys_val': 1,

 'Q_const': None,
 'Ts_const': None,
 'R_const': None,
 'sigma_const': None,
 'gamma_const': None,
 'Asys_const': None,

 'Q_prior': None,
 'Ts_prior': None,
 'R_prior': None,
 'sigma_prior': None,
 'gamma_prior': None,
 'Asys_prior': None,

 'Ts_log10': True,
 'Q_log10': False,
 'R_log10': False,
 'sigma_log10': False,
 'Asys_log10': False,
 'gamma_log10': False,

 'invert_logL': False,
 'upper_limits': False,

 # Only parameters that BubbleModel constructor understands
 'bubbles_ion': 'ion',      # or 'hot' or False
 'bubbles_pdf': 'lognormal',
 'include_rsd': 1,
 'include_mu_gt': -1,
 ###########################################################

 'restart': True,
 'burn': 0,
 'regroup_after': None,
 'steps': 100,
 'checkpoint': 10,
 'nwalkers': 128,
 'nthreads': 1,
 'suffix': None,

}

_normal = lambda x, p0, p1, p2: p0 * np.exp(-(x - p1)**2 / 2. / p2**2)

def lin_Q(Q, pars):
    return pars[0] * Q + pars[1]

def erf_Q(Q, pars):
    return pars[0] * (1. - erf(pars[1] * Q)) + pars[2]

def tanh_generic(z, pars):
    return 0.5 * (np.tanh((pars[0] - z) / pars[1]) + 1.)

def power_law(z, pars):
    return pars[0] * ((1 + z) / 8.)**pars[1]

def power_law_Q(Q, pars):
    return pars[0] * (Q / 0.5)**pars[1]

def power_law_max1(z, pars):
    return np.minimum(max_Q, power_law(z, pars))

def broken_power_law(z, pars):
    A, zb, alpha1, alpha2 = pars

    if type(z) == np.ndarray:
        lo = z <= zb
        hi = z > zb

        bpl = np.zeros_like(z)
        bpl[lo==1] = A * (z[lo==1] / zb)**alpha1
        bpl[hi==1] = A * (z[hi==1] / zb)**alpha2

    # Special case is if `pars` contains array of MCMC samples
    elif type(pars[0]) == np.ndarray:
        lo = z <= zb
        hi = z > zb

        bpl = np.zeros(pars.shape[-1])
        bpl[lo==1] = A[lo==1] * (z / zb[lo==1])**alpha1[lo==1]
        bpl[hi==1] = A[hi==1] * (z / zb[hi==1])**alpha2[hi==1]

    else:
        if z <= zb:
            bpl = A * (z / zb)**alpha1
        else:
            bpl = A * (z / zb)**alpha2

    return bpl

def broken_power_law_max1(z, pars):
    return np.minimum(max_Q, broken_power_law(z, pars))


zhigh = 20.
def double_power_law(z, pars):
    A, zpeak, alpha1, alpha2 = pars

    normcorr = 1. / ((zhigh / zpeak)**-alpha1 \
        + (zhigh / zpeak)**-alpha2)

    dpl = normcorr * A \
        * ((z / zpeak)**-alpha1 + (z / zpeak)**-alpha2)

    return dpl

def extract_params(all_pars, all_args, par):
    _args = []
    for j, _par_ in enumerate(all_pars):
        if not _par_.startswith(par):
            continue

        _args.append(all_args[j])

    return _args

class FitHelper(object):
    def __init__(self, data=None, **kwargs):
        """
        Create a class that will make setting up MCMC fits easier.

        Parameters
        ----------
        data : dict
            Must contain following keys: 'err', 'power', 'k', 'z'.
            Can provide power spectra for multiple k's, z's, and fields all at
            once. The shape of, e.g., data['power'] should be (num redshifts,
            num fields, num k modes). The same goes for data['k']. The
            dataset data['z'] should just be a 1-D array corresponding to the
            first axis of data['power'] and data['err'], while data['k'] can
            either be a 1-D array (if all modes are the same regardless of
            band or field) or a 3-D array of the same shape as data['power'].
            If fitting multiple fields at once, it is recommended to also
            pass names of the fields in data['fields'].
        kwargs : dict, optional
            Any remaining keyword arguments will be passed to the constructor
            of a micro21cm.models.BubbleModel object that generates theory
            models of the power spectrum. One can also provide additional
            arguments that help setup the fit, e.g., `nwalkers`, `steps`, etc.

        Useful functions
        ----------------
        - get_param_info: get info about model parameters.
        - get_param_dict: convert list of numbers into dictionary that can
            be understood by modeling class.
        - get_prior: return the prior value for a set of parameters.
        - get_initial_walker_pos: generate initial MCMC walker positions.
        - save_data: write results of MCMC fit to disk.
        - restart_from: get initial walker positions (and some other stuff)
            from data output so we can restart from there.

        """
        self.data = data
        self.kwargs = fit_kwargs.copy()
        self.kwargs.update(kwargs)

    @property
    def model(self):
        """
        Instance of micro21cm.models.BubbleModel that will be called in the fit.
        """
        if not hasattr(self, '_model'):
            kwargs_model = self.get_model_kwargs()
            self._model = BubbleModel(**kwargs_model)

        return self._model

    @property
    def fit_err(self):
        return self.data['err']

    @property
    def fit_fields(self):
        if not hasattr(self, '_fit_fields'):
            self._fit_fields = None
            if 'fields' in self.data:
                self._fit_fields = self.data['fields']

        return self._fit_fields

    @property
    def fit_data(self):
        if not hasattr(self, '_fit_data'):
            power = self.data['power']

            assert power.shape == (self.fit_z.size, len(self.fit_fields),
                self.fit_k.shape[2])

            self._fit_data = power

        return self._fit_data

    def get_model_kwargs(self):
        kw = self.kwargs.copy()
        return kw

    @property
    def fit_k(self):
        if not hasattr(self, '_fit_k'):
            if self.data['k'].ndim == 1:
                NzNf = np.prod(self.data['power'].shape[0:2])
                self._fit_k = np.tile(self.data['k'], NzNf).reshape(
                    self.data['power'].shape
                )
            else:
                assert self.data['k'].shape == self.data['power'].shape
                self._fit_k = self.data['k']

        return self._fit_k

    @property
    def fit_z(self):
        return self.data['z']

    def get_prefix_pars(self):
        prefix = ''
        kwargs = self.kwargs

        if kwargs['Q_func'] is not None:
            prefix += '_Q{}'.format(kwargs['Q_func'])

        if kwargs['R_func'] is not None:
            prefix += '_R{}'.format(kwargs['R_func'])

        if kwargs['Ts_func'] is not None:
            prefix += '_T{}'.format(kwargs['Ts_func'])

        if kwargs['sigma_func'] is not None:
            prefix += '_s{}'.format(kwargs['sigma_func'])
        elif (kwargs['sigma_const'] is not None) and \
            (kwargs['bubbles_pdf'][0:4] == 'logn'):
            prefix += '_sconst'

        if (kwargs['gamma_func'] is not None):
            prefix += '_g{}'.format(kwargs['gamma_func'])

        elif (kwargs['gamma_const'] is not None) and \
            (kwargs['bubbles_pdf'][0:4] == 'plex'):
            prefix += '_gconst'

        if (kwargs['Asys_func'] is not None):
            prefix += '_A{}'.format(kwargs['Asys_func'])

        return prefix

    def get_prefix_priors(self):
        prefix = ''
        kwargs = self.kwargs

        if kwargs['prior_tau']:
            prefix += '_ptau'

        if type(kwargs['prior_GP']) in [list, tuple, np.ndarray]:
            zp, Qp = kwargs['prior_GP']
            prefix += '_pGP_z{:.1f}_Q{:.2f}'.format(zp, Qp)

        return prefix

    @property
    def prefix(self):
        if not hasattr(self, '_prefix'):
            kwargs = self.kwargs
            if kwargs['bubbles_ion']:
                prefix = 'bion_{}'.format(kwargs['bubbles_pdf'][0:4])
            else:
                prefix = 'bhot_{}'.format(kwargs['bubbles_pdf'][0:4])

            prefix += self.get_prefix_pars()
            prefix += self.get_prefix_priors()

            if kwargs['suffix'] is not None:
                prefix += '_{}'.format(kwargs['suffix'])

            self._prefix = prefix

        return self._prefix

    def get_tau(self, pars):
        """
        Compute the CMB optical depth from a set of parameters.
        """
        Qofz = self._func_Q

        Y = self.model.cosmo.get_Y_p()
        y = 1. / (1. / Y - 1.) / 4.
        omega_m_0 = self.model.cosmo.omegam
        omega_b_0 = self.model.cosmo.omegab
        omega_l_0 = 1. - omega_m_0
        H0 = self.model.cosmo.H0 / km_per_mpc
        rho_crit = (3.0 * H0**2) / (8.0 * np.pi * G)
        rho_m_z0 = omega_m_0 * rho_crit
        rho_b_z0 = (omega_b_0 / omega_m_0) * rho_m_z0

        nH = lambda z: (1. - Y) * rho_b_z0 * (1. + z)**3 / m_H

        Hofz = lambda z: H0 * np.sqrt(omega_m_0 * (1.0 + z)**3  + omega_l_0)
        dldz = lambda z: c / Hofz(z) / (1. + z)

        ne = lambda z: nH(z) * Qofz(z, pars) * (1. + y)

        integrand_H_HeI = lambda z: ne(z) *  dldz(z) * sigma_T \

        # Assume helium reionization is complete at z=3
        integrand_HeII = lambda z: dldz(z) * sigma_T * nH(z) * y

        tau = quad(integrand_H_HeI, 0., 30)[0] \
            + quad(integrand_HeII, 0., 3)[0]

        return tau

    @property
    def _func_Q(self):
        if not hasattr(self, '_func_Q_'):
            self._func_Q_ = self.get_func('Q')
        return self._func_Q_

    @property
    def _func_T(self):
        if not hasattr(self, '_func_T_'):
            self._func_T_ = self.get_func('Ts')
        return self._func_T_

    @property
    def _func_R(self):
        if not hasattr(self, '_func_R_'):
            self._func_R_ = self.get_func('R')
        return self._func_R_

    @property
    def _func_sigma(self):
        if not hasattr(self, '_func_s_'):
            self._func_s_ = self.get_func('sigma')
        return self._func_s_

    @property
    def _func_gamma(self):
        if not hasattr(self, '_func_g_'):
            self._func_g_ = self.get_func('gamma')
        return self._func_g_

    @property
    def _func_A(self):
        if not hasattr(self, '_func_A_'):
            self._func_A_ = self.get_func('Asys')
        return self._func_A_

    def _func(self, par):
        if par == 'Q':
            return self._func_Q
        elif par == 'R':
            return self._func_R
        elif par == 'sigma':
            return self._func_sigma
        elif par == 'gamma':
            return self._func_gamma
        elif par == 'Ts':
            return self._func_T
        elif par == 'Asys':
            return self._func_A
        else:
            return None

    def get_func(self, par):
        name = par + '_func'
        if self.kwargs[name] is None:
            _func = None
        elif self.kwargs[name] == 'linear':
            _func = lambda Q, pars: lin_Q(Q, pars)
        elif self.kwargs[name] == 'erf':
            _func = lambda Q, pars: erf_Q(Q, pars)
        elif self.kwargs[name] == 'tanh':
            _func = lambda z, pars: tanh_generic(z, pars)
        elif self.kwargs[name] == 'pl':
            if par == 'Q':
                _func = lambda z, pars: power_law_max1(z, pars)
            elif par == 'Ts':
                _func = lambda z, pars: power_law(z, pars)
            else:
                _func = lambda Q, pars: power_law_Q(Q, pars)
        elif self.kwargs[name] == 'bpl':
            if par == 'Q':
                _func = lambda z, pars: broken_power_law_max1(z, pars)
            else:
                _func = lambda z, pars: broken_power_law(z, pars)
        elif self.kwargs[name] == 'dpl':
            _func = lambda z, pars: double_power_law(z, pars)
        else:
            raise NotImplemented('Unrecognized option for par="{}"'.format(par))

        if _func is not None and self.kwargs['{}_log10'.format(par)]:
            def func(z, pars):
                _pars = np.array(pars)
                _pars[0] = 10**pars[0]
                return _func(z, _pars)
        else:
            func = _func

        return func

    def get_priors_func(self, par_id):
        # par_id something like Q_p0, s_p0, etc.
        par, num = par_id.split('_')
        func = self.kwargs['{}_func'.format(par)]

        return _priors[par][func][num]

    def get_guess_range(self, param):
        """
        Return bounds of parameter space for i'th element in parameters list.

        .. note :: Just used for guesses! May not be the full prior range.

        See `get_param_info` for list of parameters.
        """

        params, redshifts = self.pinfo

        assert param in params, \
            "Provided `param` not in list of parameters! Options: {}".format(
            params
            )

        i = params.index(param)
        par_id = params[i]

        # Treat parameterized functions separately
        if np.isinf(redshifts[i]):
            # par_id something like Q_p0, s_p0, etc.
            par, num = par_id.split('_')
            func = self.kwargs['{}_func'.format(par)]
            lo, hi = _guesses[par][func][num]
        else:
            lo, hi = self._get_guesses_flex(i)
            num = 'p0'
            par = par_id

        if self.kwargs['{}_log10'.format(par)] and num == 'p0':
            lo = np.log10(lo)
            hi = np.log10(hi)

        return lo, hi

    def _get_guesses_flex(self, i):
        params, redshifts = self.pinfo
        par = params[i]

        # Potentially change default guess range to user-supplied prior.
        if self.kwargs['{}_prior'.format(par)] is not None:
            lo, hi = self.kwargs['{}_prior'.format(par)]
        else:
            lo, hi = _guesses[par]['broad']

        return lo, hi

    @property
    def nparametric(self):
        if not hasattr(self, '_num_parametric'):
            num = 0
            for par in self.model.params:
                num += self.kwargs['{}_func'.format(par)] is not None

            self._num_parametric = num

        return self._num_parametric

    @property
    def nparams(self):
        N = 0
        for par in self.model.params:
            if self.kwargs['{}_val'.format(par)] is not None:
                continue

            # Parameter not allowed to evolve with redshift.
            if self.kwargs['{}_const'.format(par)] is not None:
                if self.kwargs['{}_const'.format(par)]:
                    N += 1
                    continue

            func = self.kwargs['{}_func'.format(par)]
            is_func = func is not None
            if is_func:
                if func in ['pl', 'tanh']:
                    N += 2
                elif func in ['bpl', 'dpl']:
                    N += 4
                elif func in ['erf']:
                    N += 3
                elif func in ['lin', 'linear']:
                    N += 2
            else:
                N += self.fit_z.size

        return N

    @property
    def pinfo(self):
        return self.get_param_info()

    def get_initial_walker_pos(self):
        """
        Generate a set of initial walker positions that randomly and uniformly
        sample the prior volume.

        Returns
        -------
        An array with dimension (num walkers, num parameters).
        """

        nwalkers = self.kwargs['nwalkers']
        params, redshifts = self.pinfo

        pos = np.zeros((nwalkers, self.nparams))
        for i, par in enumerate(params):
            lo, hi = self.get_guess_range(par)
            pos[:,i] = lo + np.random.rand(nwalkers) * (hi - lo)

        return pos

    def get_param_info(self):
        """
        Figure out mapping from parameter list to parameter names and redshifts.

        Returns
        -------
        Tuple containing (parameter names, redshifts for each parameter). Note
        that if a parameter does not correspond to a single redshift, e.g.,
        because it is a component of a parametric function, then the
        redshift element will be -np.inf.
        """

        ct = 0
        param_z = []
        param_names = []
        for i, _z_ in enumerate(self.fit_z):

            for j, par in enumerate(self.model.params):
                if self.kwargs['{}_val'.format(par)] is not None:
                    continue
                if self.kwargs['{}_func'.format(par)] is not None:
                    continue

                # If par is non-evolving, only add once
                if self.kwargs['{}_const'.format(par)] is not None:
                    if par in param_names:
                        continue

                param_z.append(_z_)
                param_names.append(par)

        # If parameterizing Q or R, these will be at the end.
        if self._func_Q is not None:
            if self.kwargs['Q_func'] in ['tanh', 'pl']:
                param_z.extend([-np.inf]*2)
                param_names.extend(['Q_p0', 'Q_p1'])
            elif self.kwargs['Q_func'] == 'bpl':
                param_z.extend([-np.inf]*4)
                param_names.extend(['Q_p0', 'Q_p1', 'Q_p2', 'Q_p3'])
            else:
                raise NotImplementedError('Unrecognized function {}'.format(
                    self.kwargs['Q_func']
                ))

        if self._func_T is not None:
            assert self.kwargs['Ts_func'] in ['dpl', 'pl']

            if self.kwargs['Ts_func'] == 'dpl':
                param_z.extend([-np.inf]*4)
                param_names.extend(['Ts_p0', 'Ts_p1', 'Ts_p2',
                    'Ts_p3'])
            elif self.kwargs['Ts_func'] == 'pl':
                param_z.extend([-np.inf]*2)
                param_names.extend(['Ts_p0', 'Ts_p1'])

        if self._func_R is not None:
            assert self.kwargs['R_func'] == 'pl'
            param_z.extend([-np.inf]*2)
            param_names.extend(['R_p0', 'R_p1'])

        if self._func_sigma is not None:
            assert self.kwargs['sigma_func'] in ['pl', 'linear']
            param_z.extend([-np.inf]*2)
            param_names.extend(['sigma_p0', 'sigma_p1'])

        if self._func_A is not None:
            assert self.kwargs['Asys_func'] == 'erf'
            param_z.extend([-np.inf]*3)
            param_names.extend(['Asys_p0', 'Asys_p1', 'Asys_p2'])

        return param_names, param_z

    def restart_from(self, fn):
        """
        Read previous output and generate new positions for walkers.

        Returns
        -------
        Tuple containing (position of all walkers, dictionary containing
        full dataset read from previous output, random state for emcee).
        """

        f = open(fn, 'rb')
        data_pre = pickle.load(f)
        f.close()

        kwargs = self.kwargs

        nwalkers_p, steps_p, pars_p = data_pre['chain'].shape
        warning = 'Number of walkers must match to enable restart!'
        warning += 'Previous run ({}) used {}'.format(fn, nwalkers_p)
        assert nwalkers_p == kwargs['nwalkers'], warning
        print("% Restarting from output {}.".format(fn))
        print("% Will augment {} samples/walker there with {} more (per walker).".format(
            steps_p, kwargs['checkpoint']))

        rstate = data_pre['rstate']

        # Set initial walker positions
        pos = None
        if kwargs['regroup_after']:
            # Re-initialize all walkers to best points after some number of steps
            if data_pre['chain'].shape[1] == kwargs['regroup_after']:
                print("Re-positioning walkers around 10% best so far...")
                # Take top 10% of walkers?
                nw = kwargs['nwalkers']
                numparam = data_pre['chain'].shape[-1]
                top = int(0.1 * nw)


                # Most recent step for all walkers
                ibest = np.argsort(data_pre['lnprob'][:,-1])[-top:]

                pos = np.zeros((nw, data_pre['chain'].shape[-1]))
                for i in range(nw):
                    j = ibest[i % top]
                    # Add some random jitter?
                    r = 1. + np.random.normal(scale=0.05, size=numparam)
                    pos[i,:] = data_pre['chain'][j,-1,:] * r


                pos = np.array(pos)


        if pos is None:
            pos = data_pre['chain'][:,-1,:]

        return pos, data_pre, rstate

    def save_data(self, fn, sampler, data_pre=None):
        """
        Write data out to file with name `fn`.

        Parameters
        ----------
        fn : str
            Filename of output.
        sampler : object
            An emcee.EnsembleSampler object.
        data_pre : dict
            Data contained in last save file.

        """
        ##
        # Write data
        # micro21cm.inference.write_chain(sampler, data_pre)
        if data_pre is not None:
            chain = data_pre['chain']
            fchain = data_pre['flatchain']
            lnprob = data_pre['lnprob']
            blobs = np.array(data_pre['blobs'])
            facc = np.array(data_pre['facc'])

            if not np.allclose(self.fit_k, data_pre['kblobs']):
                raise ValueError("k-bins used in previous fit are different!")

            # Happens if we only took one step before
            #if blobs.ndim == 2:
            #    blobs = np.array([blobs])

            # chain is (nwalkers, nsteps, nparams)
            # blobs is (nsteps, nwalkers, nredshifts, nkbins)
            _sblobs = sampler.blobs
            sblobs = _sblobs if _sblobs.ndim == 4 else np.expand_dims(_sblobs, 2)

            data = {'chain': np.concatenate((chain, sampler.chain), axis=1),
                'flatchain': np.concatenate((fchain, sampler.flatchain)),
                'lnprob': np.concatenate((lnprob, sampler.lnprobability), axis=1),
                'blobs': np.concatenate((blobs, sblobs)),
                'facc': np.concatenate((facc,
                    np.array(sampler.acceptance_fraction))),
                'kbins': self.fit_k, 'kblobs': self.fit_k,
                'zfit': self.fit_z, 'data': self.fit_data,
                'pinfo': self.pinfo, 'rstate': sampler.random_state,
                'kwargs': self.kwargs}
        else:
            if sampler.blobs.ndim == 3:
                # Should only need to do this once.
                blobs = np.expand_dims(sampler.blobs, 2)
            else:
                blobs = np.array(sampler.blobs)

            data = {'chain': sampler.chain, 'flatchain': sampler.flatchain,
                'lnprob': sampler.lnprobability,
                'blobs': blobs,
                'facc': sampler.acceptance_fraction,
                'kbins': self.fit_k, 'kblobs': self.fit_k,
                'zfit': self.fit_z, 'data': self.fit_data,
                'pinfo': self.pinfo, 'rstate': sampler.random_state,
                'kwargs': self.kwargs}

        with open(fn, 'wb') as f:
            pickle.dump(data, f)

        print("% Wrote {} at {}.".format(fn, time.ctime()))

    def get_param_dict(self, z, args, ztol=1e-3):
        """
        Take complete list of parameters from emcee and make dictionary
        of parameters for a single redshift.

        .. note :: If Q(z) is parameterized, will do that calculation here
            automatically.

        """
        model = self.model

        allpars, redshifts = self.get_param_info()

        pars = {}
        for i, par in enumerate(model.params):

            if self.kwargs['{}_func'.format(par)] is not None:
                _args = extract_params(allpars, args, par)

                if par in ['Q', 'Ts']:
                    pars[par] = self._func(par)(z, _args)

                    if par == 'Q':
                        Q_of_z = pars[par]
                else:
                    pars[par] = self._func(par)(Q_of_z, _args)

            elif self.kwargs['{}_val'.format(par)] is not None:
                pars[par] = self.kwargs['{}_val'.format(par)]
            elif self.kwargs['{}_const'.format(par)] is not None:
                j = allpars.index(par)
                pars[par] = args[j]
            else:
                pok = np.zeros(len(allpars))
                zok = np.zeros(len(allpars))
                for k, element in enumerate(redshifts):
                    if element is None:
                        continue

                    if abs(redshifts[k] - z) < ztol:
                        zok[k] = 1
                    if allpars[k] == par:
                        pok[k] = 1

                ok = np.logical_and(pok, zok)

                if ok.sum() == 0:
                    return None

                assert ok.sum() == 1

                j = int(np.argwhere(ok==1))

                if self.kwargs['{}_log10'.format(par)]:
                    pars[par] = 10**args[j]
                else:
                    pars[par] = args[j]

        if pars == {}:
            return None

        return pars

    def get_prior_range(self, param):
        """
        Return the prior range for parameter `param`.
        """

        params, redshifts = self.get_param_info()

        assert param in params, \
            "Provided `param` not in list of parameters! Options: {}".format(
            params
            )

        i = params.index(param)

        # Treat parameterized functions separately
        if np.isinf(redshifts[i]):
            par, num = param.split('_')
            lo, hi = self.get_priors_func(param)
        else:
            lo, hi = _priors[param]['broad']
            num = 'p0'
            par = param

        # Allow user to override internal defaults.
        # [do here to get 'par' in case we're parameterized]
        if self.kwargs['{}_prior'.format(par)] is not None:
            lo, hi = self.kwargs['{}_prior'.format(par)]

        if self.kwargs['{}_log10'.format(par)] and num == 'p0':
            lo = np.log10(lo)
            hi = np.log10(hi)

        return lo, hi

    def get_prior(self, args):
        """
        Get prior for input set of parameters.

        Parameters
        ----------
        args : list, tuple, np.ndarray
            Single set of parameter values to be evaluated. Should be 1-D,
            and be of length `self.nparams`.

        """

        params, redshifts = self.get_param_info()

        for i, par_id in enumerate(params):

            lo, hi = self.get_prior_range(par_id)

            if not (lo <= args[i] <= hi):
                return -np.inf

        ##
        # Check for priors on Q(z=late in reionization)
        if type(self.kwargs['prior_GP']) in [list, tuple, np.ndarray]:
            zp, Qp = self.kwargs['prior_GP']
            Qpars = extract_params(params, args, 'Q')
            if self._func_Q(zp, Qpars) < Qp:
                return -np.inf

        ##
        # Check for priors on tau
        lnP = 0.
        if self.kwargs['prior_tau'] not in [None, False, 0]:
            # Assume Gaussian
            Qpars = extract_params(params, args, 'Q')
            tau = self.get_tau(Qpars)
            lnP -= np.log(_normal(tau, 1., 0.055, 0.009))

        # If we made it this far, everything is OK
        return lnP
