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

fig, axes = pl.subplots(1, 3, figsize=(15,5))

R = np.logspace(-2, 3, 1000)
k = 1. / R

model = micro21cm.BubbleModel()

cf_mm = micro21cm.util.get_cf_from_ps(R, lambda kk: model.get_ps_matter(7.9, kk))
ps_mm = model.get_ps_matter(7.9, k)

# Compare PS and CF
axes[0].loglog(k, ps_mm)
axes[2].loglog(R, np.abs(cf_mm))

# Recover PS from CF
_fcf = interp1d(np.log(R), cf_mm, kind='cubic', bounds_error=False, fill_value=0)
f_cf = lambda RR: _fcf.__call__(np.log(RR))

ps_rec = micro21cm.util.get_ps_from_cf(k, f_cf, Rmin=R.min(), Rmax=R.max())
axes[0].loglog(k, ps_rec, ls='--', label='recovered')
axes[0].legend()

axes[1].loglog(k, ps_mm * k**3 / 2. / np.pi**2)
axes[1].loglog(k, ps_rec * k**3 / 2. / np.pi**2, ls='--', label='recovered')
