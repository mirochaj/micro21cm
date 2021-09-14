"""

test_models_bsd.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Thu 17 Dec 2020 10:40:05 EST

Description:

"""

import micro21cm
import numpy as np
import matplotlib.pyplot as pl

def test():

    z = 8.
    k = np.logspace(-1, 0, 5)

    model_logn = micro21cm.BubbleModel(bubbles_pdf='lognormal')
    model_plex = micro21cm.BubbleModel(bubbles_pdf='plexp')

    bsd_logn = model_logn.get_bsd(Q=0.1, R=2., sigma=1)
    bsd_plex = model_plex.get_bsd(Q=0.1, R=2., gamma=-3.5)

    # Test R vs. Rpeak
    R = model_logn.get_R_from_Rpeak(Q=0.1, R=2., sigma=1)
    Rp = model_logn.get_Rpeak_from_R(Q=0.1, R=R, sigma=1)
    assert Rp == 2., "{} {}".format(R, Rp)

    R = model_plex.get_R_from_Rpeak(Q=0.1, R=2., sigma=1)
    Rp = model_plex.get_Rpeak_from_R(Q=0.1, R=R, sigma=1)
    assert Rp == 2., "{} {}".format(R, Rp)

    # Make sure variance is bigger when smoothing on smaller scale.
    assert model_logn.get_variance_bb(z, 1., Q=0.1, R=2., sigma=1) \
         > model_logn.get_variance_bb(z, 8., Q=0.1, R=2., sigma=1)

    ps_bb = model_logn.get_ps_bb(z, k, Q=0.1, R=2., sigma=1)
    ps_bd = model_logn.get_ps_bd(z, k, Q=0.1, R=2., sigma=1)

    assert np.all(ps_bb > ps_bd)

if __name__ == '__main__':
    test()
