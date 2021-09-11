"""

test_models_ps_components.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Thu 17 Dec 2020 16:25:39 EST

Description:

"""

import micro21cm
import numpy as np

kw = {'R': 5., 'sigma': 1., 'Q': 0.5}

def test():

    model = micro21cm.BubbleModel()

    P1 = [model.get_P1(RR, **kw) for RR in model.tab_R]
    P2 = [model.get_P2(RR, **kw) for RR in model.tab_R]

    # Test limiting behaviour. Should this work better?
    assert np.allclose(kw['Q'], P1[0]+P2[0], rtol=0, atol=1e-3)
    assert np.allclose(kw['Q']**2, P1[-1]+P2[-1], rtol=0, atol=1e-3)

if __name__ == '__main__':
    test()
