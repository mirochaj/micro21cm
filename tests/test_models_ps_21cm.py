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
    model_nob = micro21cm.BubbleModel(bubbles=False)

    # Set modes of interest
    z = 8.
    k = np.logspace(-1., 0, 5)

    # Compute P(k)
    Tcmb = model.get_Tcmb(z)

    ps_pre = np.zeros(k.size)
    for Ts in [0.5 * Tcmb, 0.1 * Tcmb]:

        ps = model.get_ps_21cm(z=z, k=k, Ts=Ts, Q=0.2, R=3., sigma=1)
        assert np.all(ps > ps_pre), "{} {}".format(ps, ps_pre)
        ps_pre = ps

        ps_nob = model_nob.get_ps_21cm(z=z, k=k, Ts=Ts, Q=0.2, R=3., sigma=1)
        assert np.all(ps > ps_nob)

    # Check ability to calibrate to known PS
    kw = model.calibrate_ps(k, k**3 * ps / 2. / np.pi**2, Q=0.2, z=z,
        which_ps='21cm', R=3., sigma=1., xtol=1e-2)

    assert abs(kw['Ts'] - Ts) < 1e-1, kw['Ts']


if __name__ == '__main__':
    test()
