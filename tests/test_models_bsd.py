"""

test_models_bsd.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Thu 17 Dec 2020 10:40:05 EST

Description:

"""

import time
import pytest
import micro21cm
import numpy as np
import matplotlib.pyplot as pl

@pytest.mark.parametrize('use_bmf,', [(True,), (False,)])
def test(use_bmf):

    z = 8.
    k = np.logspace(-1, 0, 5)

    model_logn = micro21cm.BubbleModel(bubbles_pdf='lognormal',
        normalize_via_bmf=use_bmf)
    model_n = micro21cm.BubbleModel(bubbles_pdf='normal',
        normalize_via_bmf=use_bmf)
    model_plex = micro21cm.BubbleModel(bubbles_pdf='plexp',
        normalize_via_bmf=use_bmf)
    model_nob = micro21cm.BubbleModel(bubbles=False)

    # Check alpha
    assert model_logn.get_alpha(z, Ts=np.inf) == -1
    assert model_nob.get_alpha(z, Ts=np.inf) == 0

    model_hot = micro21cm.BubbleModel(bubbles_pdf='lognormal', bubbles_ion=0)
    assert model_hot.get_alpha(z, Ts=10.) not in [0, -1]

    # Check mean dTb
    assert model_logn.get_dTb_avg(z, Q=0.0) == model_logn.get_dTb_bulk(z=z)

    # Check BSDs
    t1 = time.time()
    bsd_logn = model_logn.get_bsd(Q=0.1, R=2., sigma=1)
    t2 = time.time()
    t3 = time.time()
    bsd_logn2 = model_logn.get_bsd(Q=0.1, R=2., sigma=1)
    t4 = time.time()

    assert (t4 - t3) < (t2 - t1), "Check caching."

    # Check mass function call
    bmf = model_logn.get_bmf(z, Q=0.1, R=2., sigma=1)

    # Gaussian BSD -- don't really use this but check it.
    bsd_n1 = model_n.get_bsd(Q=0.1, R=2., sigma=1)
    bsd_n2 = model_n.get_bsd(Q=0.1, R=2., sigma=1, alpha=2)

    # Check that we get 0 if Q=0
    bb = model_logn.get_bb(z, Q=0)
    assert np.allclose(bb, 0)

    # Check limiting behaviour
    bb1, bb2 = model_logn.get_bb(z, Q=0.1, separate=True)
    assert np.allclose(bb2[-1] , 0.1**2), bb2[-1]

    # Check caching
    assert (t4-t3) < (t2-t1), \
        "Caching should speed things up! {} {}".format(t4-t3, t2-t1)
    assert np.all(bsd_logn == bsd_logn2)

    # Check skewed log-normal
    bsd_logn_sk_p = model_logn.get_bsd(Q=0.1, R=2., sigma=1, alpha=1)
    bsd_logn_sk_m = model_logn.get_bsd(Q=0.1, R=2., sigma=1, alpha=-1)

    bsd_plex = model_plex.get_bsd(Q=0.1, R=2., gamma=-3.5)

    # Test R vs. Rpeak
    R = model_logn.get_R_from_Rpeak(Q=0.1, R=2., sigma=1)
    Rp = model_logn.get_Rpeak_from_R(Q=0.1, R=R, sigma=1)
    assert Rp == 2., "{} {}".format(R, Rp)

    assert model_logn.get_Rpeak(Q=0.1, R=2.) == 2.

    R = model_plex.get_R_from_Rpeak(Q=0.1, R=2., sigma=1)
    Rp = model_plex.get_Rpeak_from_R(Q=0.1, R=R, sigma=1)
    assert Rp == 2., "{} {}".format(R, Rp)

    # Make sure variance is bigger when smoothing on smaller scale.
    assert model_logn.get_variance_bb(z, 1., Q=0.1, R=2., sigma=1) \
         > model_logn.get_variance_bb(z, 8., Q=0.1, R=2., sigma=1)

    # Make sure we can get PS of things we usually only deal with via CFs
    # (until the very end when we go from CF_21cm to PS_21cm)
    ps_bb = model_logn.get_ps_bb(z, k, Q=0.1, R=2., sigma=1)
    ps_bd = model_logn.get_ps_bd(z, k, Q=0.1, R=2., sigma=1)

    assert np.all(ps_bb > ps_bd)

    # Make sure we recover the input characteristic bubble size
    #kw = model_logn.calibrate_ps(k, k**3 * ps_bb / 2. / np.pi**2,
    #    Q=0.1, z=z, which_ps='bb', xtol=1e-2, sigma=1., free_R=True)
    #assert (abs(kw['R'] - 2.) < 1e-1), 'Recovered R={}'.format(kw['R'])

    # Check some convenience functions
    cdf = model_logn.get_bsd_cdf(Q=0.1, R=2., sigma=1)
    R_med = model_logn.tab_R[np.argmin(np.abs(cdf - 0.5))]
    nb = model_logn.get_nb(Q=0.1, R=2., sigma=1)

    rofk = model_logn.get_r_of_k(z, k, Q=0.1, R=2., sigma=1)

    assert np.all(rofk > 0)
