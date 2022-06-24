"""

test_inference_multiz.py

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
    sys_argv = ['scriptname', 'steps=1', 'checkpoint=1', 'nwalkers=16',
        'prior_tau=True', 'bubbles_pdf=lognormal', 'Ts_prior=[0,20]',
        'Ts_log10=False', 'prior_GP=[5.3,0.99]', 'Q_func=pl', 'Ts_func=pl',
        'R_func=pl', 'sigma_const=1', 'Asys_val=1',
        'nonsense=hello_123']

    kwargs = micro21cm.inference.fit_kwargs.copy()
    kwargs.update(micro21cm.get_cmd_line_kwargs(sys_argv))

    data = {}
    data['k'] = np.array([0.2, 0.5])
    data['fields'] = ['A']
    data['z'] = np.array([8., 10.])
    data['power'] = np.array([[[100., 200.]], [[50., 200.]]])
    data['err'] = np.array([[[10., 20.]], [[10., 20.]]])

    # dummy (noise, cosmic variance)
    def get_error(z, k):
        return 10. * np.ones_like(k), np.zeros_like(k)

    helper = micro21cm.inference.FitHelper(data, get_error, **kwargs)

    # Do an actual model here to test more machinery.
    def get_ps(z, k, **kw):
        return helper.model.get_ps_21cm(z, k, **kw) * k**3 / 2. / np.pi**2

    par_list = ['sigma', 'Q_p0', 'Q_p1', 'Ts_p0', 'Ts_p1', 'R_p0', 'R_p1']

    assert helper.nparams == 7, helper.pinfo[0]
    assert helper.fit_z.size == 2
    assert helper.tab_k.size == 2
    assert helper.pinfo[0] == par_list, helper.pinfo[0]
    assert helper.num_parametric == 3
    assert helper.func_sigma == None
    assert helper.func('sigma') is None
    assert helper.func_gamma == None
    assert helper.func('gamma') is None

    pos = helper.get_initial_walker_pos()

    assert pos.shape == (helper.kwargs['nwalkers'], helper.nparams)

    pars8 = helper.get_param_dict(z=8, args=pos[0])
    pars10 = helper.get_param_dict(z=10, args=pos[0])

    for _par_ in ['Q', 'R']:
        assert pars8[_par_] > pars10[_par_]

    def loglikelihood(pars):

        blobs = -np.inf * np.ones((helper.fit_z.size, helper.tab_k.size))

        # Get prior (unenforced in this test to make sure model is called)
        lnP = helper.get_prior(pars)

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
    fn = '{}.pkl'.format(helper.prefix)
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
