"""

test_models_ps_21cm.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import micro21cm
import numpy as np

def test():
    model = micro21cm.BubbleModel()

    # Set modes of interest
    z = 8
    k = np.logspace(-1., 0, 5)

    # Compute P(k)
    Tcmb = model.get_Tcmb(z)

    ps_pre = np.zeros(k.size)
    for Ts in [0.5 * Tcmb, 0.1 * Tcmb]:

        ps = model.get_ps_21cm(z=z, k=k, Ts=Ts, Q=0.5, R=3., sigma=1)
        assert np.all(ps > ps_pre)
        ps_pre = ps 


if __name__ == '__main__':
    test()
