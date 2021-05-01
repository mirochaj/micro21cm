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
import emcee
import pickle
import numpy as np
from scipy.special import erf

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    from schwimmbad import MPIPool
except ImportError:
    rank = 0
    size = 1

_priors = \
{
 'Ts': (0., 1000.),
 'Q': (0, 1),
 'R_b': (0, 30),
 'sigma_b': (0.05, 1.0),
}

_guesses = \
{
 'Ts': (5., 150.),
 'Q': (0.2, 0.8),
 'R_b': (0.5, 10.),
 'sigma_b': (0.2, 0.8),
}

_guesses_Q_tanh = {'p0': (6, 10), 'p1': (1, 4)}
_guesses_R_pl = {'p0': (2., 4.), 'p1': (-2., -1.)}

_priors_Q_tanh = {'p0': (5, 15), 'p1': (0, 20)}
_priors_R_pl = {'p0': (0, 30), 'p1': (-4, 0.)}

fit_kwargs = \
{
 'fit_z': 5, # If None, fits all redshifts!
 'Qprior': True,
 'Rprior': True,
 'Qxdelta': True, # Otherwise, vary *increments* in Q.
 'Rxdelta': True, # Otherwise, vary *increments* in Q.
 'Qfunc': None,
 'Rfunc': None,

 'kmax': 1,
 'kthin': None,

 'bubbles_ion': 'ion',      # or 'hot' or False
 'bubbles_pdf': 'lognormal',
 'include_rsd': 2,
 'Rfree': True,

 'restart': True,
 'regroup_after': None,
 'steps': 10,
 'nwalkers': 128,
 'nthreads': 1,
 'suffix': None,

}

Q_stagger = lambda z: 1. - min(1, max(0, (z - 5.) / 5.))
R_stagger = lambda z: ((z - 5.) / 5.)**-2

def tanh_generic(z, zref, dz):
    return 0.5 * (np.tanh((zref - z) / dz) + 1.)

def power_law(z, norm, gamma):
    return norm * ((1 + z) / 8.)**gamma


class FitHelper(object):
    def __init__(self, data, **kwargs):
        self.data = data
        self.kwargs = fit_kwargs.copy()
        self.kwargs.update(kwargs)

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
    def fit_zindex(self):
        if not hasattr(self, '_fit_z'):
            fit_z = self.kwargs['fit_z']

            if type(fit_z) in [list, tuple, np.ndarray]:
                pass
            else:
                fit_z = np.array([fit_z])

            self._fit_z = np.array(fit_z, dtype=int)

        return self._fit_z

    def get_z_from_index(self, i, data):
        return data['z'][i]

    @property
    def prefix(self):
        if not hasattr(self, '_prefix'):
            kwargs = self.kwargs
            if kwargs['bubbles_ion']:
                prefix = 'bion_{}'.format(kwargs['bubbles_pdf'][0:4])
            else:
                prefix = 'bhot_{}'.format(kwargs['bubbles_pdf'][0:4])
            #else:
            #    prefix = 'beq0'

            s_prior = ''

            if kwargs['Qfunc'] is not None:
                prefix += '_Q{}'.format(kwargs['Qfunc'])
            else:
                if kwargs['Qxdelta']:
                    prefix += '_vdQ'

                if kwargs['Qprior']:
                    s_prior += 'Qinc'

            if kwargs['Rfunc'] is not None:
                prefix += '_R{}'.format(kwargs['Rfunc'])
            else:
                if kwargs['Rfree']:
                    prefix += '_vRb'
                else:
                    prefix += '_vnb'

                if kwargs['Rxdelta']:
                    prefix += '_vdR'

                if kwargs['Rprior']:
                    s_prior += 'Rinc'

            prefix += '_mock_21cmfast_{}'.format(kwargs['mocknum'])

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

    @property
    def func_Q(self):
        if not hasattr(self, '_func_Q'):
            if self.kwargs['Qfunc'] is None:
                self._func_Q = None
            elif self.kwargs['Qfunc'] == 'tanh':
                self._func_Q = lambda z, zref, dz: tanh_generic(z, zref, dz)
            else:
                raise NotImplemented('help')

        return self._func_Q

    @property
    def func_R(self):
        if not hasattr(self, '_func_R'):
            if self.kwargs['Rfunc'] is None:
                self._func_R = None
            elif self.kwargs['Rfunc'] == 'pl':
                self._func_R = lambda z, norm, index: power_law(z, norm, index)
            else:
                raise NotImplemented('help')

        return self._func_R



    @property
    def num_parametric(self):
        if not hasattr(self, '_num_parametric'):
            self._num_parametric = (self.kwargs['Qfunc'] is not None) \
                + (self.kwargs['Rfunc'] is not None)
        return self._num_parametric

    def nparams(self, model):
        N = (len(model.params) - self.num_parametric) * self.fit_zindex.size
        N += self.num_parametric * 2
        return N

    def get_initial_walker_pos(self, model):
        nwalkers = self.kwargs['nwalkers']

        pos = np.zeros((nwalkers, self.nparams(model)))

        params, redshifts = self.get_param_info(model)

        for i, par in enumerate(params):

            # If parameterized, be careful
            if redshifts[i] is None:
                if par.startswith('Q'):
                    post = par[2:]
                    lo, hi = _guesses_Q_tanh[post]
                elif par.startswith('R'):
                    post = par[2:]
                    lo, hi = _guesses_R_pl[post]
                else:
                    raise NotImplemented('help')
            else:
                # Can parameterize change in Q, R_b, rather than Q, R_b
                # themselves.
                if (par == 'Q') and self.kwargs['Qxdelta']:
                    lo, hi = 0, 0.2
                elif par == 'Q':
                    Q0 = Q_stagger(redshifts[i])
                    dQ = 1. / float(len(self.fit_zindex) - 1)

                    lo = Q0 - 0.5 * dQ
                    hi = Q0 + 0.5 * dQ
                elif (par == 'R_b') and self.kwargs['Rxdelta']:
                    lo, hi = 0, 2
                elif par == 'R_b':
                    R0 = R_stagger(redshifts[i])
                    dR = 1. / float(len(self.fit_zindex) - 1)

                    lo = R0 - 0.5 * dR
                    hi = R0 + 0.5 * dR
                else:
                    lo, hi = _guesses[par]

            pos[:,i] = lo + np.random.rand(nwalkers) * (hi - lo)

        return pos

    def get_param_info(self, model):
        """
        Figure out mapping from parameter list to parameter names and redshifts.
        """

        par_per_z = len(model.params)
        if self.func_Q is not None:
            par_per_z -= 1
        if self.func_R is not None:
            par_per_z -= 1

        ct = 0
        param_z = []
        param_names = []
        for i, iz in enumerate(self.fit_zindex):
            _z_ = self.get_z_from_index(iz, self.data)

            for j, par in enumerate(model.params):

                if (par == 'Q') and (self.func_Q is not None):
                    continue
                if (par == 'R_b') and (self.func_R is not None):
                    continue

                param_z.append(_z_)
                param_names.append(par)


        # If parameterizing Q or R, these will be at the end.
        if self.func_Q is not None:
            param_z.extend([None]*2)
            param_names.extend(['Q_p0', 'Q_p1'])
        if self.func_R is not None:
            param_z.extend([None]*2)
            param_names.extend(['R_p0', 'R_p1'])

        return param_names, param_z

    def get_param_dict(self, model, z, args, ztol=1e-3):
        """
        Take complete list of parameters from emcee and make dictionary
        of parameters for a single redshift.

        .. note :: If Q(z) is parameterized, will do that calculation here
            automatically.

        """

        allpars, redshifts = self.get_param_info(model)




        pars = {}
        for i, par in enumerate(model.params):

            # Check for parametric options
            if (par == 'Q') and (self.func_Q is not None):
                pars['Q'] = self.func_Q(z, args[-4], args[-3])
            elif (par == 'R_b') and (self.func_R is not None):
                pars['R_b'] = self.func_R(z, args[-2], args[-1])
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

                pars[par] = args[j]

        if pars == {}:
            return None

        return pars

    def get_prior(self, model, args):

        params, redshifts = self.get_param_info(model)

        Q = []
        R = []
        for i, par in enumerate(params):
            if redshifts[i] is None:
                if par.startswith('Q'):
                    post = par[2:]
                    lo, hi = _priors_Q_tanh[post]
                elif par.startswith('R'):
                    post = par[2:]
                    lo, hi = _guesses_R_pl[post]
                else:
                    raise NotImplemented('help')
            else:
                # Can parameterize change in Q, R_b, rather than Q, R_b
                # themselves.
                lo, hi = _priors[par]

            if par == 'Q':
                Q.append(args[i])
            elif par == 'R':
                R.append(args[i])

            if not (lo <= args[i] <= hi):
                return -np.inf


        if self.kwargs['Qprior'] and (self.func_Q is None):
            if not np.all(np.diff(Q) < 0):
                return -np.inf
        if self.kwargs['Rprior'] and (self.func_R is None):
            if not np.all(np.diff(R) < 0):
                return -np.inf


        # If we made it this far, everything is OK
        return 0.
