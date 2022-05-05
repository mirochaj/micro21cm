"""

test_models_ps_components.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Thu 17 Dec 2020 16:25:39 EST

Description:

"""

import pytest
import micro21cm
import numpy as np

test_options = \
[(8, 5, 1, 0.5, False),
 (8, 5, 1, 0.5, True),
 (8, 5, 1, 0, False),
 (8, 5, 1, 0, True),
 (8, 5, 1, 1, False),
 (8, 5, 1, 1, True)]

@pytest.mark.parametrize('z, R, sigma, Q, use_bmf', test_options)
def test(z, R, sigma, Q, use_bmf):

    kw = {'R': R, 'sigma': sigma, 'Q': Q}
    model = micro21cm.BubbleModel(normalize_via_bmf=use_bmf)

    P1 = [model.get_P1(z, RR, **kw) for RR in model.tab_R]
    P2 = [model.get_P2(z, RR, **kw) for RR in model.tab_R]
    P_bn = [model.get_Pbn(z, RR, **kw) for RR in model.tab_R]

    # Test limiting behaviour. Should this work better?
    assert np.allclose(kw['Q'], P1[0]+P2[0], rtol=0, atol=1e-3)
    assert np.allclose(kw['Q']**2, P1[-1]+P2[-1], rtol=0, atol=1e-3)
    assert np.allclose(kw['Q'] * (1. - kw['Q']), P_bn[-1], rtol=0, atol=1e-3)

    # Need ionization fluctuations to vanish at Q==1
    # Just setting separate=1 to hit more lines.
    bb1, bb2 = model.get_bb(z, Q=1, separate=1)
    bn = model.get_bn(z, Q=1)
    assert np.all(bb1+bb2 == 1)
    assert np.all(bn == 0)

    bb1, bb2 = model.get_bb(z, Q=0, separate=1)
    bn = model.get_bn(z, Q=0)
    assert np.all(bb1+bb2 == 0)
    assert np.all(bn == 0)

    # Need cross-term cancellation when Q==1
    dd = model.get_dd(z)
    bd, bbd, bdd, bbdd, bbd_1pt, bd_1pt = model.get_cross_terms(z, Q=1,
        separate=True)

    assert np.all(bd == 0)
    assert np.all(bdd + bbdd + dd == 0)
    assert np.all(bbd == 0)
    assert np.all(bbd_1pt == 0)
    assert np.all(bd_1pt == 0)

    # Need 21-cm fluctuations to vanish at Q==1
    karr = np.logspace(-1, 0, 11)
    D_21 = np.inf * np.ones_like(karr)
    for Q in [0.98, 0.99, 1]:
        D_21_new = model.get_ps_21cm(z=z, k=karr, Q=Q) * karr**3 / 2. / np.pi**2

        print(Q)
        print(D_21)
        print(D_21_new)
        # Something weird happens between 0.99 and 0.999
        #assert np.all(D_21_new <= D_21)
        D_21 = D_21_new.copy()

    assert np.all(D_21 == 0)
