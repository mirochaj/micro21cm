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
 'R': (0, 30),
 'sigma': (0.05, 2),
 'gamma': (-5, 4),
}

_guesses_broad = \
{
 'Ts': (10., 1000.),
 'Q': (0.2, 0.8),
 'R': (0.5, 10.),
 'sigma': (0.6, 1.5),
 'gamma': (-4, -2),
}

_bins = \
{
 'Ts': np.arange(0, 500, 10),
 'Q': np.arange(-0.01, 1.01, 0.01),
 'R': np.arange(0, 50, 0.2),
 'sigma': np.arange(0.0, 1.0, 0.05),
}

_guesses_Q_tanh = {'p0': (6, 10), 'p1': (1, 4)}
_guesses_Q_bpl = {'p0': (0.25, 0.75), 'p1': (6, 10), 'p2': (-3, -1),
    'p3': (-4, -2)}

_guesses_Q = {'tanh': _guesses_Q_tanh, 'bpl': _guesses_Q_bpl,
    'broad': _guesses_broad['Q']}

_guesses_R_pl = {'p0': (1., 5.), 'p1': (-10., -20.)}
_guesses_R = {'pl': _guesses_R_pl, 'broad': _guesses_broad['R']}

_guesses_T_dpl = {'p0': (5., 15.), 'p1': (8, 20), 'p2': (1, 3),
    'p3': (-2.5, -1.5)}
_guesses_T = {'broad': _guesses_broad['Ts'], 'dpl': _guesses_T_dpl}

_guesses_s_pl = {'p0': (0.3, 0.6), 'p1': (-0.5, 0.5)}
_guesses_s = {'pl': _guesses_s_pl, 'broad': _guesses_broad['sigma']}

_guesses_g = {'broad': _guesses_broad['gamma']}

_guesses = {'R': _guesses_R, 'Q': _guesses_Q, 'Ts': _guesses_T,
    'sigma': _guesses_s, 'gamma': _guesses_g}

_priors_Q_tanh = {'p0': (5, 15), 'p1': (0, 20)}
_priors_Q_bpl = {'p0': (0, 1), 'p1': (5, 20), 'p2': (-6, 0), 'p3': (-6, 0)}
_priors_Q = {'tanh': _priors_Q_tanh, 'bpl': _priors_Q_bpl,
    'broad': _priors_broad['Q']}

_priors_R_pl = {'p0': (0, 30), 'p1': (-25, -1)}
_priors_R = {'pl': _priors_R_pl, 'broad': _priors_broad['R']}

_priors_s_pl = {'p0': (0.0, 1), 'p1': (-2, 2)}

_priors_s = {'pl': _priors_s_pl, 'broad': _priors_broad['sigma']}

_priors_T_dpl = {'p0': (0, 50), 'p1': (5, 30), 'p2': (0, 6), 'p3': (-6, 0)}
_priors_T = {'broad': _priors_broad['Ts'], 'dpl': _priors_T_dpl}
_priors_g = {'broad': _priors_broad['gamma']}

_priors = {'Q': _priors_Q, 'R': _priors_R, 'sigma': _priors_s,
    'Ts': _priors_T, 'gamma': _priors_g}

fit_kwargs = \
{
 'fit_z': 0, # If None, fits all redshifts!

 'prior_tau': False,
 'prior_GP': False,
 'Qxdelta': False, # Otherwise, vary *increments* in Q.
 'Rxdelta': False, # Otherwise, vary *increments* in Q.
 'Q_func': None,
 'R_func': None,
 'Ts_func': None,
 'sigma_func': None,
 'gamma_func': None,
 'Q_monotonic': True,
 'R_monotonic': True,
 'Ts_monotonic': False,
 'sigma_monotonic': False,
 'gamma_monotonic': False,

 'Ts_log10': True,

 'kmin': 0.1,
 'kmax': 1,
 'kthin': None,
 'invert_logL': False,
 'upper_limits': False,

 'bubbles_ion': 'ion',      # or 'hot' or False
 'bubbles_pdf': 'lognormal',
 'include_rsd': 2,
 'Rfree': True,

 'restart': True,
 'regroup_after': None,
 'steps': 100,
 'checkpoint': 10,
 'nwalkers': 128,
 'nthreads': 1,
 'suffix': None,

}

Q_stagger = lambda z: 1. - min(1, max(0, (z - 5.) / 5.))
R_stagger = lambda z: ((z - 5.) / 5.)**-2

_normal = lambda x, p0, p1, p2: p0 * np.exp(-(x - p1)**2 / 2. / p2**2)

def tanh_generic(z, pars):
    return 0.5 * (np.tanh((pars[0] - z) / pars[1]) + 1.)

def power_law(z, pars):
    return pars[0] * ((1 + z) / 8.)**pars[1]

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
    def __init__(self, data=None, data_err_func=None, **kwargs):
        self.data = data
        self.data_err_func = data_err_func
        self.kwargs = fit_kwargs.copy()
        self.kwargs.update(kwargs)

    @property
    def model(self):
        if not hasattr(self, '_model'):
            kwargs_model = self.get_model_kwargs()
            self._model = BubbleModel(**kwargs_model)

        return self._model

    @property
    def fit_data(self):
        if not hasattr(self, '_fit_data'):

            data_to_fit = []
            for h, _z_ in enumerate(self.fit_z):
                _data = self.data['power'][self.get_zindex_in_data(_z_)]

                # Should apply window function here
                if self.data_err_func is not None:
                    _noise, _a21 = self.data_err_func(_z_, self.tab_k)
                    ydat = _data['Deltasq21'][self.k_mask==0]
                    perr = _data['errDeltasq21'][self.k_mask==0]
                    yerr = _noise + _a21 * ydat + perr
                else:
                    ydat = _data['Deltasq21']
                    yerr = _data['errDeltasq21']

                data_to_fit.append([ydat, yerr])

            self._fit_data = np.array(data_to_fit)

        return self._fit_data

    def get_model_kwargs(self):
        kw = {}
        if self.kwargs['bubbles_ion']:
            kw['bubbles_ion'] = True
        else:
            kw['bubbles_ion'] = False

        kw['bubbles_pdf'] = self.kwargs['bubbles_pdf']
        kw['include_rsd'] = self.kwargs['include_rsd']
        kw['bubbles_Rfree'] = self.kwargs['Rfree']

        return kw

    @property
    def tab_k(self):
        if not hasattr(self, '_kblobs'):
            kblobs = self.data['power'][0]['k']
            kmin = self.kwargs['kmin']
            kmax = self.kwargs['kmax']

            k_ok = np.logical_and(kblobs >= kmin, kblobs <= kmax)
            kblobs = kblobs[k_ok==1]

            if self.kwargs['kthin'] is not None:
                kblobs = kblobs[::int(self.kwargs['kthin'])]

            self._kblobs = kblobs
            mask = np.ones(self.data['power'][0]['k'].size, dtype=int)
            for i, k in enumerate(self.data['power'][0]['k']):
                if k not in kblobs:
                    continue

                mask[i] = 0

            self._k_mask = mask

        return self._kblobs

    @property
    def k_mask(self):
        if not hasattr(self, '_k_mask'):
            k = self.tab_k
        return self._k_mask

    @property
    def fit_zindex(self):
        if not hasattr(self, '_fit_zindex'):
            fit_z = self.kwargs['fit_z']

            if type(fit_z) in [list, tuple, np.ndarray]:
                pass
            else:
                fit_z = np.array([fit_z])

            self._fit_zindex = np.array(fit_z, dtype=int)

        return self._fit_zindex

    def get_z_from_index(self, i):
        return self.data['z'][i]

    def get_zindex_in_data(self, z, ztol=1e-3):
        j = np.argmin(np.abs(z - self.data['z']))
        assert abs(self.data['z'][j] - z) < ztol

        return j

    @property
    def fit_z(self):
        if not hasattr(self, '_fit_z'):
            self._fit_z = np.array([self.get_z_from_index(i) \
                for i in self.fit_zindex])
        return self._fit_z

    @property
    def prefix(self):
        if not hasattr(self, '_prefix'):
            kwargs = self.kwargs
            if kwargs['bubbles_ion']:
                prefix = 'bion_{}'.format(kwargs['bubbles_pdf'][0:4])
            else:
                prefix = 'bhot_{}'.format(kwargs['bubbles_pdf'][0:4])

            s_prior = ''

            if kwargs['Q_func'] is not None:
                prefix += '_Q{}'.format(kwargs['Q_func'])
            else:
                if kwargs['Qxdelta']:
                    prefix += '_vdQ'

                if kwargs['Q_monotonic']:
                    s_prior += 'Qmon'

            if kwargs['R_func'] is not None:
                prefix += '_R{}'.format(kwargs['R_func'])
            else:
                if kwargs['Rfree']:
                    prefix += '_vR'
                else:
                    prefix += '_vnb'

                if kwargs['Rxdelta']:
                    prefix += '_vdR'

                if kwargs['R_monotonic']:
                    s_prior += 'Rmon'

            if kwargs['Ts_func'] is not None:
                prefix += '_T{}'.format(kwargs['Ts_func'])

            if kwargs['sigma_func'] is not None:
                prefix += '_s{}'.format(kwargs['sigma_func'])

            if kwargs['prior_tau']:
                prefix += '_ptau'

            if type(kwargs['prior_GP']) in [list, tuple, np.ndarray]:
                zp, Qp = kwargs['prior_GP']
                prefix += '_pGP_z{:.1f}_Q{:.2f}'.format(zp, Qp)

            if kwargs['fit_z'] is None:
                prefix += '_zall'
                if kwargs['Qprior'] or kwargs['Rprior']:
                    prefix += '_' + s_prior
            elif type(kwargs['fit_z']) in [list, tuple, np.ndarray]:
                s = ''
                for iz in kwargs['fit_z']:
                    s += str(int(iz))

                prefix += '_z{}'.format(s)
                if s_prior.strip():
                    prefix += '_' + s_prior
            else:
                prefix += '_z{}'.format(kwargs['fit_z'])

            if kwargs['kmax'] is not None:
                prefix += '_kmax_{:.1f}'.format(kwargs['kmax'])
            if kwargs['kthin'] is not None:
                prefix += '_kthin_{:.0f}'.format(kwargs['kthin'])

            if kwargs['suffix'] is not None:
                prefix += '_{}'.format(kwargs['suffix'])

            self._prefix = prefix

        return self._prefix

    def get_tau(self, pars):
        Qofz = self.func_Q

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
    def func_Q(self):
        if not hasattr(self, '_func_Q'):
            self._func_Q = self.get_func('Q')
        return self._func_Q

    @property
    def func_T(self):
        if not hasattr(self, '_func_T'):
            self._func_T = self.get_func('Ts')
        return self._func_T

    @property
    def func_R(self):
        if not hasattr(self, '_func_R'):
            self._func_R = self.get_func('R')
        return self._func_R

    @property
    def func_sigma(self):
        if not hasattr(self, '_func_s'):
            self._func_s = self.get_func('sigma')
        return self._func_s

    @property
    def func_gamma(self):
        if not hasattr(self, '_func_g'):
            self._func_s = self.get_func('gamma')
        return self._func_s

    def func(self, par):
        if par == 'Q':
            return self.func_Q
        elif par == 'R':
            return self.func_R
        elif par == 'sigma':
            return self.func_s
        elif par == 'gamma':
            return self.func_gamma
        elif par == 'Ts':
            return self.func_T
        else:
            return None

    def get_func(self, par):
        name = par + '_func'
        if self.kwargs[name] is None:
            func = None
        elif self.kwargs[name] == 'tanh':
            func = lambda z, pars: tanh_generic(z, pars)
        elif self.kwargs[name] == 'pl':
            func = lambda z, pars: power_law_max1(z, pars)
        elif self.kwargs[name] == 'bpl':
            func = lambda z, pars: broken_power_law_max1(z, pars)
        elif self.kwargs[name] == 'dpl':
            func = lambda z, pars: double_power_law(z, pars)
        else:
            raise NotImplemented('help')

        return func

    def get_priors_func(self, par_id):
        # par_id something like Q_p0, s_p0, etc.
        par, num = par_id.split('_')
        func = self.kwargs['{}_func'.format(par)]

        return _priors[par][func][num]

    def get_guesses_func(self, i):
        params, redshifts = self.pinfo
        par_id = params[i]

        # par_id something like Q_p0, s_p0, etc.
        par, num = par_id.split('_')

        func = self.kwargs['{}_func'.format(par)]

        return _guesses[par][func][num]

    def get_guesses_flex(self, i):
        params, redshifts = self.pinfo
        par = params[i]

        # Can parameterize change in Q, R, rather than Q, R
        # themselves.
        if (par == 'Q') and self.kwargs['Qxdelta']:
            lo, hi = 0, 0.2
        elif par == 'Q':
            if self.fit_z.size == 1:
                lo, hi = 0, 1
            else:
                Q0 = Q_stagger(redshifts[i])
                dQ = 1. / float(max(len(self.fit_zindex) - 1, 1))

                lo = Q0 - 0.5 * dQ
                hi = Q0 + 0.5 * dQ
        elif (par == 'R') and self.kwargs['Rxdelta']:
            lo, hi = 0, 2
        elif par == 'R':
            R0 = R_stagger(redshifts[i])
            dR = 1. / float(max(len(self.fit_zindex) - 1, 1))

            lo = R0 - 0.5 * dR
            hi = R0 + 0.5 * dR
        else:
            lo, hi = _guesses[par]['broad']

        return lo, hi

    @property
    def num_parametric(self):
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
            func = self.kwargs['{}_func'.format(par)]
            is_func = func is not None
            if is_func:
                if func in ['pl', 'tanh']:
                    N += 2
                elif func in ['bpl', 'dpl']:
                    N += 4
            else:
                N += self.fit_zindex.size

        return N

    @property
    def pinfo(self):
        return self.get_param_info()

    def get_initial_walker_pos(self):
        nwalkers = self.kwargs['nwalkers']

        pos = np.zeros((nwalkers, self.nparams))

        params, redshifts = self.pinfo

        for i, par in enumerate(params):

            # If parameterized, be careful
            if np.isinf(redshifts[i]):
                lo, hi = self.get_guesses_func(i)
            else:
                lo, hi = self.get_guesses_flex(i)

            if par == 'Ts' and self.kwargs['Ts_log10']:
                lo = np.log10(lo)
                hi = np.log10(hi)

            pos[:,i] = lo + np.random.rand(nwalkers) * (hi - lo)

        return pos

    def get_param_info(self):
        """
        Figure out mapping from parameter list to parameter names and redshifts.
        """

        ct = 0
        param_z = []
        param_names = []
        for i, iz in enumerate(self.fit_zindex):
            _z_ = self.get_z_from_index(iz)

            for j, par in enumerate(self.model.params):

                if self.kwargs['{}_func'.format(par)] is not None:
                    continue

                param_z.append(_z_)
                param_names.append(par)

        # If parameterizing Q or R, these will be at the end.
        if self.func_Q is not None:
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

        if self.func_T is not None:
            assert self.kwargs['Ts_func'] == 'dpl'
            param_z.extend([-np.inf]*4)
            param_names.extend(['Ts_p0', 'Ts_p1', 'Ts_p2', 'Ts_p3'])

        if self.func_R is not None:
            assert self.kwargs['R_func'] == 'pl'
            param_z.extend([-np.inf]*2)
            param_names.extend(['R_p0', 'R_p1'])

        if self.func_sigma is not None:
            assert self.kwargs['sigma_func'] == 'pl'
            param_z.extend([-np.inf]*2)
            param_names.extend(['sigma_p0', 'sigma_p1'])

        return param_names, param_z

    def restart_from(self, fn):
        """
        Read previous output and generate new positions for walkers.
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

        return pos, data_pre

    def save_data(self, fn, sampler, data_pre=None):
        ##
        # Write data
        # micro21cm.inference.write_chain(sampler, data_pre)
        if data_pre is not None:
            chain = data_pre['chain']
            fchain = data_pre['flatchain']
            lnprob = data_pre['lnprob']
            blobs = np.array(data_pre['blobs'])
            facc = np.array(data_pre['facc'])

            if not np.allclose(self.tab_k, data_pre['kblobs']):
                raise ValueError("k-bins used in previous fit are different!")

            # Happens if we only took one step before
            if blobs.ndim == 2:
                blobs = np.array([blobs])

            # chain is (nwalkers, nsteps, nparams)
            # blobs iss (nsteps, nwalkers, nparams)
            data = {'chain': np.concatenate((chain, sampler.chain), axis=1),
                'flatchain': np.concatenate((fchain, sampler.flatchain)),
                'lnprob': np.concatenate((lnprob, sampler.lnprobability), axis=1),
                'blobs': np.concatenate((blobs, np.array(sampler.blobs))),
                'facc': np.concatenate((facc,
                    np.array(sampler.acceptance_fraction))),
                'kbins': self.tab_k, 'kblobs': self.tab_k,
                'zfit': self.fit_z, 'data': self.fit_data,
                'pinfo': self.pinfo,
                'kwargs': self.kwargs}
        else:
            data = {'chain': sampler.chain, 'flatchain': sampler.flatchain,
                'lnprob': sampler.lnprobability,
                'blobs': np.array(sampler.blobs),
                'facc': sampler.acceptance_fraction,
                'kbins': self.tab_k, 'kblobs': self.tab_k,
                'zfit': self.fit_z, 'data': self.fit_data,
                'pinfo': self.pinfo,
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
                pars[par] = self.func(par)(z, _args)

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

                if par == 'Ts' and self.kwargs['Ts_log10']:
                    pars[par] = 10**args[j]
                else:
                    pars[par] = args[j]

        if pars == {}:
            return None

        return pars

    def get_prior(self, args):

        model = self.model
        params, redshifts = self.get_param_info()

        parevol = {par: [] for par in self.model.params}
        for i, par_id in enumerate(params):
            if np.isinf(redshifts[i]):
                lo, hi = self.get_priors_func(par_id)
            else:
                par = par_id

                if par == 'Ts' and self.kwargs['Ts_log10']:
                    lo, hi = np.log10(_priors[par]['broad'])
                else:
                    lo, hi = _priors[par]['broad']

            parevol[par].append(args[i])

            if not (lo <= args[i] <= hi):
                return -np.inf

        for par in self.model.params:
            if not self.kwargs['{}_monotonic'.format(par)]:
                continue
            if (self.kwargs['{}_func'.format(par)]) is not None:
                continue
            if not np.all(np.diff(parevol[par]) < 0):
                return -np.inf

        ##
        # Check for priors on Q(z=late in reionization)
        if type(self.kwargs['prior_GP']) in [list, tuple, np.ndarray]:
            zp, Qp = self.kwargs['prior_GP']
            Qpars = extract_params(params, args, 'Q')
            if self.func_Q(zp, Qpars) < Qp:
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
