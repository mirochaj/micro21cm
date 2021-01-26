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
    have_mpi = True
except ImportError:
    rank = 0
    size = 1
    have_mpi = False

_default_priors = \
{
 'Ts': (0., 100.),
 'Q': (0, 1),
 'R_b': (0, 30),
 'sigma_b': (0.05, 0.5),
}

_default_guesses = \
{
 'Ts': (1., 4.),
 'Q': (0.2, 0.8),
 'R_b': (1., 5.),
 'sigma_b': (0.15, 0.3),
}

def _get_ps(z, k, model, **kwargs):
    ps_21 = model.get_ps_21cm(z=z, k=k, **kwargs)
    D_21 = ps_21 * k**3 / 2. / np.pi**2
    return D_21

def likelihood(data, model, err, limits, invert):

    if limits:
        # Likelihood for upper limit with unknown systematic
        if invert:
            _l_ = -1. * (data - model) / np.sqrt(2.) / err
        else:
            _l_ = (data - model) / np.sqrt(2.) / err

        L = 0.5 * np.sqrt(np.pi) * (1. + erf(_l_))
    else:
        raise NotImplemented('Only dealing with upper limits so far.')
        #lnL = -0.5 * (np.sum((ymod - ydat)**2 \
        #        / yerr**2 + np.log(2. * np.pi * yerr**2)))

    return L

def loglikelihood(pars, *args):
    """
    Assumes parameter ordering is: R_b, sigma_b, Q, Ts
    Assumes args ordering is: ion or heat, k modes, invert likelihood

    """

    model, data, kblobs, invert, priors = args

    pars_dict = {}
    for i, par in enumerate(model.params):
        pars_dict[par] = pars[i]

    Nz = len(data.keys())
    zdata = np.sort(list(data.keys()))

    if zdata.size > 1 and rank == 0:
        print("WARNING: multi-z fits aren't worthwhile at this stage.")

    if kblobs is not None:
        blobs_buff = -np.inf * np.ones((zdata.size, len(kblobs)))
    else:
        blobs_buff = {}

    # Hard-coded priors (sorry)
    for par in model.params:
        if not (priors[par][0] <= pars_dict[par] <= priors[par][1]):
            return -np.inf, blobs_buff

    ##
    # Enforce priors
    for par in pars:
        if par < 0:
            return -np.inf, blobs_buff

    # Loop over contents of `data`
    L = []
    blobs = blobs_buff.copy()
    for i, z in enumerate(zdata):
        _data = data[z]
        _kdat = _data['k']

        ymod = _get_ps(z, _kdat, model, **pars_dict)
        ydat = _data['D_sq']
        yerr = _data['err']

        # Standard likelihood
        #lnL = -0.5 * (np.sum((ymod - ydat)**2 \
        #        / yerr**2 + np.log(2. * np.pi * yerr**2)))

        # Likelihood for upper limit with unknown systematic
        #if invert:
        #    _l_ = -1. * (ydat - ymod) / np.sqrt(2.) / yerr
        #else:
        #    _l_ = (ydat - ymod) / np.sqrt(2.) / yerr

        _l_ = likelihood(ydat, ymod, yerr, True, invert)

        if kblobs is not None:
            blobs[i,:] = _get_ps(z, kblobs, model, **pars_dict)

        L.extend(0.5 * np.sqrt(np.pi) * (1. + erf(_l_)))

    lnL = np.sum(np.log(L))

    return lnL, blobs


class DummyEnsembleSampler(object):
    def __init__(self):
        pass


class FitGrid(object):
    def __init__(self, data, model, kblobs=None, invert_logL=True):
        self.data = data
        self.model = model
        self.kblobs = kblobs
        self.invert_logL = invert_logL

        assert 'k' not in self.data.keys(), \
            "Format of `data` should be dict of dicts (one per redshift)."

class FitMCMC(object):
    def __init__(self, data, model, kblobs=None, invert_logL=True):
        self.data = data
        self.model = model
        self.kblobs = kblobs
        self.invert_logL = invert_logL

        assert 'k' not in self.data.keys(), \
            "Format of `data` should be dict of dicts (one per redshift)."

    @property
    def priors(self):
        if not hasattr(self, '_priors'):
            self._priors = _default_priors.copy()
        return self._priors

    def get_initial_walker_pos(self, nwalkers):
        pos = np.zeros((nwalkers, len(self.model.params)))

        for i, par in enumerate(self.model.params):
            lo, hi = _default_guesses[par]
            pos[:,i] = lo + np.random.rand(nwalkers) * (hi - lo)

        return pos

    def run_fit(self, steps=100, nwalkers=64, nthreads=1, prefix=None,
        restart=False):
        """
        Run emcee.
        """

        args = [self.model, self.data, self.kblobs, self.invert_logL,
            self.priors]

        Nparams = len(self.model.params)

        if (nthreads == 1) and (size > 1):
            print("% WARNING: Not sure MPIPool works yet.")

            pool = MPIPool()

            if not pool.is_master():
                pool.wait()
                sys.exit(0)
        else:
            pool = None

        ##
        # Check for previous output
        pos = None
        data_pre = None
        if prefix is not None:
            fn = prefix + '.pkl'
            if os.path.exists(fn) and restart:
                f = open(fn, 'rb')
                data_pre = pickle.load(f)
                f.close()

                nwalkers_p, steps_p, pars_p = data_pre['chain'].shape

                warning = 'Number of walkers must match to enable restart!'
                warning += 'Previous run ({}) used {}'.format(fn, nwalkers_p)
                assert nwalkers_p == nwalkers, warning

                print("% Restarting from output {}.".format(fn))
                print("% Will augment {} samples there with {} more.".format(
                    steps_p, steps
                ))

                # Set initial walker positions
                pos = data_pre['chain'][:,-1,:]

                # Or, use sample_ball to re-initialize

        # Initial guesses for four parameters: R_b, sigma_b, Q, Ts
        if pos is None:
            pos = self.get_initial_walker_pos(nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers, Nparams, loglikelihood,
            threads=nthreads, args=args, pool=pool)

        print("% Starting MCMC at {}".format(time.ctime()))

        t1 = time.time()
        results = sampler.run_mcmc(pos, steps)
        t2 = time.time()

        print("% Ran MCMC for {} steps ({:.1f} minutes).".format(steps,
            (t2 - t1) / 60.))

        if pool is not None:
            pool.close()    

        ##
        # Optionally save
        if prefix is not None:

            if data_pre is not None:
                chain = data_pre['chain']
                fchain = data_pre['flatchain']

                data = {'chain': np.concatenate((chain, sampler.chain), axis=1),
                    'flatchain': np.concatenate((fchain, sampler.flatchain))}
            else:
                data = {'chain': sampler.chain, 'flatchain': sampler.flatchain}

            with open(fn, 'wb') as f:
                pickle.dump(data, f)

            print("% Wrote {}.".format(fn))

        return sampler
