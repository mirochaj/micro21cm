"""

test_inference_singlez.py

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

    # Make sure parametric functions work
    xarr = np.arange(0, 1, 0.1)
    zarr = np.arange(6, 11, 1)
    micro21cm.inference.lin_Q(xarr, [0.2, 1])
    micro21cm.inference.erf_Q(xarr, [0.5, 0.2, 0.0])
    micro21cm.inference.tanh_generic(zarr, [8, 2])
    micro21cm.inference.power_law(zarr, [10., 2])
    micro21cm.inference.power_law_Q(xarr, [10, 2])
    micro21cm.inference.power_law_max1(zarr, [0.1, -2])
    micro21cm.inference.broken_power_law(zarr, [1, 10, -2, 2])
    micro21cm.inference.broken_power_law(9., [1, 10, -2, 2])
    micro21cm.inference.broken_power_law(9.,
        np.array([1, 10, -2, 2]*10).reshape(4, 10))
    micro21cm.inference.broken_power_law_max1(zarr, [1, 10, -2, 2])
    micro21cm.inference.double_power_law(zarr, [1, 10, -2, 2])

    ##
    # Actually run a (very simple) fit.

    # fake command-line arguments
    sys_argv = ['scriptname', 'steps=3', 'checkpoint=1', 'nwalkers=10',
        'prior_tau=False', 'bubbles_pdf=lognormal', 'Ts_prior=[0,20]',
        'Ts_log10=False', 'sigma_val=None', 'regroup_after=2']

    kwargs = micro21cm.inference.fit_kwargs.copy()
    kwargs.update(micro21cm.get_cmd_line_kwargs(sys_argv))

    data = {}
    data['k'] = np.array([0.2, 0.5])
    data['fields'] = ['A']
    data['z'] = np.array([8.])
    data['power'] = np.array([[[100., 200.]]])
    data['err'] = np.array([[[10., 20.]]])

    # dummy (noise, cosmic variance)
    def get_error(z, k):
        return 10. * np.ones_like(k), np.zeros_like(k)

    helper = micro21cm.inference.FitHelper(data, **kwargs)

    # Doesn't have to be model call here, make dumber just for speed.
    def get_ps(z, k, **kw):
        #return model.get_ps_21cm(z, k, **kw) * k**3 / 2. / np.pi**2
        return kw['Ts'] * ((1. + z) / 8.) * (1. - kw['Q'])

    # Changed default to Asys_val=1 so nparams should be 4
    assert helper.nparams == 4, helper.pinfo[0]
    assert helper.fit_z.size == 1
    assert helper.fit_k.size == 2
    assert helper.pinfo[0] == ['Q', 'Ts', 'R', 'sigma'], helper.pinfo[0]
    assert np.isfinite(helper.get_prior([0.1, 10., 5., 1.]))

    pars = helper.get_param_dict(z=8, args=[0.1, 10., 5., 1.])

    def loglikelihood(pars):

        blobs = -np.inf * np.ones_like(helper.fit_data)

        lnL = None
        for h, _z_ in enumerate(helper.fit_z):

            pars_dict = helper.get_param_dict(_z_, pars)

            ymod = get_ps(_z_, data['k'], **pars_dict)

            # Single field, hence [0]
            ydat = data['power'][h][0]
            perr = data['err'][h][0]

            _noise, _a21 = get_error(_z_, data['k'])
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
    fn = '{}.pkl'.format(helper.prefix)
    Ncheckpts = kwargs['steps'] // kwargs['checkpoint'] # just to test restart
    for i in range(Ncheckpts):
        # Set initial positions of walkers
        if (kwargs['restart'] and os.path.exists(fn)) and (i > 0):
            pos, data_pre, rstate = helper.restart_from(fn)
        else:
            data_pre = None
            rstate = None
            pos = helper.get_initial_walker_pos()

        # Initialize sampler
        sampler = emcee.EnsembleSampler(kwargs['nwalkers'], helper.nparams,
            loglikelihood, pool=None)
        sampler.random_state = rstate

        # Run it
        results = sampler.run_mcmc(pos, kwargs['checkpoint'])

        # Write data
        helper.save_data(fn, sampler, data_pre)
