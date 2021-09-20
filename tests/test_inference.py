"""

test_inference.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 21:31:49 EST

Description:

"""

import os
import emcee
import micro21cm
import numpy as np

def test():
    # fake command-line arguments
    sys_argv = ['scriptname', 'steps=2', 'checkpoint=1', 'nwalkers=8',
        'prior_tau=False', 'bubbles_pdf=lognormal']

    kwargs = micro21cm.inference.fit_kwargs.copy()
    kwargs.update(micro21cm.get_cmd_line_kwargs(sys_argv))

    data = {'z': np.array([8.])}
    data_z8 = \
    {
     'k': np.array([0.2, 0.5]),
     'Deltasq21': np.array([100., 200.]),
     'errDeltasq21': np.array([10., 20.]),
    }
    data['power'] = [data_z8]

    # dummy (noise, cosmic variance)
    def get_error(z, k):
        return 10. * np.ones_like(k), np.zeros_like(k)

    helper = micro21cm.inference.FitHelper(data, get_error, **kwargs)

    # Doesn't have to be model call here, make dumber just for speed.
    def get_ps(z, k, **kw):
        #return model.get_ps_21cm(z, k, **kw) * k**3 / 2. / np.pi**2
        return kw['Ts'] * ((1. + z) / 8.) * (1. - kw['Q'])

    assert helper.nparams == 4
    assert helper.fit_z.size == 1
    assert helper.tab_k.size == 2
    assert helper.pinfo[0] == ['Ts', 'Q', 'R', 'sigma']

    def loglikelihood(pars):

        blobs = -np.inf * np.ones((helper.fit_z.size, helper.tab_k.size))

        lnL = None
        for h, _z_ in enumerate(helper.fit_z):

            _data = data['power'][h]

            pars_dict = helper.get_param_dict(_z_, pars)

            ymod = get_ps(_z_, _data['k'], **pars_dict)

            ydat = _data['Deltasq21'][helper.k_mask==0]
            perr = _data['errDeltasq21'][helper.k_mask==0]

            _noise, _a21 = get_error(_z_, _data['k'])
            yerr = _noise + _a21 * ydat + perr

            _lnL = np.sum(-0.5 * (ydat - ymod)**2 / 2. / yerr**2)
            if lnL is None:
                lnL = _lnL * 1.
            else:
                lnL += _lnL

            blobs[h,:] = ymod

        if np.isnan(lnL):
            return -np.inf, blobs

        return lnL, blobs

    ##
    # Actually run
    fn = 'test_fit.pkl'
    Ncheckpts = kwargs['steps'] // kwargs['checkpoint'] # just to test restart
    for i in range(Ncheckpts):
        # Set initial positions of walkers
        if (kwargs['restart'] and os.path.exists(fn)) and (i > 0):
            pos, data_pre = helper.restart_from(fn)
        else:
            data_pre = None
            pos = helper.get_initial_walker_pos()

        # Initialize sampler
        sampler = emcee.EnsembleSampler(kwargs['nwalkers'], helper.nparams,
            loglikelihood, pool=None)

        # Run it
        results = sampler.run_mcmc(pos, kwargs['checkpoint'])

        # Write data
        helper.save_data(fn, sampler, data_pre)

if __name__ == '__main__':
    test()
