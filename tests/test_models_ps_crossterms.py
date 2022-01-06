"""

test_models_ps_crossterms.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Fri 27 Aug 2021 14:52:12 EDT

Description:

"""

import micro21cm
import numpy as np

z = 8
Ts = 3.
R = 5
Q = 0.4
sigma = 1
#kwargs = dict(Q=0.25, R_b=2., sigma_b=0.5, n_b=None, Ts=Ts)
kwargs = dict(Q=Q, R=R, sigma=sigma, Ts=Ts)
setup_kw = {'use_volume_match': 1}
karr = np.logspace(-1, 0., 41)

def test():

    model = micro21cm.BubbleModel(**setup_kw)
    model_3 = micro21cm.BubbleModel(include_cross_terms=3, **setup_kw)
    model_bin = micro21cm.BubbleModel(include_cross_terms=2, **setup_kw)
    model_1 = micro21cm.BubbleModel(include_cross_terms=1, **setup_kw)
    model_ideal = micro21cm.BubbleModel(include_cross_terms=0, **setup_kw)

    bd, bbd, bdd, bbdd, bbd_1pt, bd_1pt = \
        model_bin.get_cross_terms(z, separate=True, **kwargs)
    other = bbd + bdd + bbdd + bbd_1pt + bd_1pt
    all_terms = model_bin.get_cross_terms(z, separate=False, **kwargs)
    assert np.allclose(other+bd, all_terms)

    all_terms3 = model_3.get_cross_terms(z, separate=False, **kwargs)
    all_terms2 = model_ideal.get_cross_terms(z, separate=False, **kwargs)
    all_terms1 = model_1.get_cross_terms(z, separate=False, **kwargs)

    dd = model.get_dd(z)
    bb = model.get_bb(z, **kwargs)
    bn = model.get_bn(z, **kwargs)
    d_i = model.get_bubble_density(z, **kwargs)

    # Check limiting behaviours
