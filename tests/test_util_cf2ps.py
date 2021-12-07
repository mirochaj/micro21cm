"""

test_util_cf2ps.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import time
import micro21cm
import numpy as np
from scipy.interpolate import interp1d

def test(rtol=1e-2):

    R = np.logspace(-2, 3, 1000)
    z = 8.
    k = 1. / R

    model = micro21cm.BubbleModel()

    # Get matter PS
    ps_func = lambda kk: model.get_ps_matter(z, kk)

    # Compute CF from PS using Clenshaw-Curtis integration
    cf = micro21cm.util.get_cf_from_ps_func(R, ps_func)
    ps = model.get_ps_matter(z, k)

    # Recover PS from CF using Clenshaw-Curtis again
    _fcf = interp1d(np.log(R), cf, kind='cubic', bounds_error=False,
        fill_value=0)
    f_cf = lambda RR: _fcf.__call__(np.log(RR))
    ps_cc = micro21cm.util.get_ps_from_cf_func(k, f_cf, Rmin=R.min(),
        Rmax=R.max())
    err_cc = np.abs(ps - ps_cc) / ps

    # Recover CF from PS using mcfit method, then retrieve PS from that CF.
    # i.e, use mcfit both ways
    R_mc, cf_mc = micro21cm.util.get_cf_from_ps_tab(k, ps)
    k_mc, ps_mc = micro21cm.util.get_ps_from_cf_tab(R_mc, cf_mc)
    # Interpolate to common k's to calculate error
    ps_mci = np.exp(np.interp(np.log(k), np.log(k_mc), np.log(ps_mc)))

    err_mc = np.abs(ps - ps_mci) / ps
    err_cf = np.abs(cf - cf_mc) / cf

    # Recover PS from Clenshaw-Curtis-generated-CF using mcfit, i.e.,
    # just use mcfit for one way.
    k_mc1, ps_mc1 = micro21cm.util.get_ps_from_cf_tab(R, cf)
    ps_mci1 = np.exp(np.interp(np.log(k), np.log(k_mc1), np.log(ps_mc1)))
    # Interpolate again
    err_mc1 = np.abs(ps - ps_mci1) / ps

    # Assess accuracy over k interval we care about most
    kok = np.logical_and(k >= 1e-1, k <= 1)
    assert np.all(err_cc[kok==1] < rtol)
    assert np.all(err_mc[kok==1] < rtol)
    assert np.all(err_mc1[kok==1] < rtol)


if __name__ == '__main__':
    test()
