"""

test_util_cf2ps.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import micro21cm
import numpy as np
import matplotlib.pyplot as pl
from scipy.interpolate import interp1d

def test():

    R = np.logspace(-2, 3, 1000)
    z = 8.
    k = 1. / R

    model = micro21cm.BubbleModel()

    cf_mm = micro21cm.util.get_cf_from_ps(R, lambda kk: model.get_ps_matter(z, kk))
    ps_mm = model.get_ps_matter(z, k)

    # Recover PS from CF
    _fcf = interp1d(np.log(R), cf_mm, kind='cubic', bounds_error=False,
        fill_value=0)
    f_cf = lambda RR: _fcf.__call__(np.log(RR))

    ps_rec = micro21cm.util.get_ps_from_cf(k, f_cf, Rmin=R.min(), Rmax=R.max())
    

if __name__ == '__main__':
    test()
