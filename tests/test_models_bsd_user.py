"""

test_models_bsd_user.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Fri 30 Jul 2021 16:52:59 EDT

Description:

"""

import micro21cm
import numpy as np
import matplotlib.pyplot as pl

def test():

    model = micro21cm.BubbleModel()

    z = 8
    k = 0.2
    kw = {'Q': 0.4, 'R': 5, 'sigma': 1, 'Ts': 10}

    R = model.tab_R
    bsd = model.get_bsd(**kw)

    ps_in = model.get_ps_21cm(z=z, k=k, **kw)

    # Initialize model
    model = micro21cm.BubbleModel(bubbles_pdf=(R, bsd))

    ps = model.get_ps_21cm(z=z, k=k, **kw)

    assert np.allclose(ps_in, ps)

if __name__ == '__main__':
    test()
