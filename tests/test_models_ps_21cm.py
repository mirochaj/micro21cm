"""

test_models_ps_21cm.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import micro21cm
import numpy as np
import matplotlib.pyplot as pl

def test():
    model = micro21cm.BubbleModel()

    # Set modes of interest
    k = np.logspace(-1., 0, 21)

    # Compute P(k)
    ps = model.get_ps_21cm(z=8., k=k, Ts=3., Q=0.5, R=3., sigma=1)

    # Plot dimensionless power spectrum
    pl.loglog(k, k**3 * ps / 2. / np.pi**2)

    pl.savefig('{!s}.png'.format(__file__[0:__file__.rfind('.')]))
    pl.close()

if __name__ == '__main__':
    test()
