"""

test_models_ps_components.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Thu 17 Dec 2020 16:25:39 EST

Description:

"""

import micro21cm
import numpy as np
import matplotlib.pyplot as pl

zplot = 8.
Qplot = 0.1
kw = {'R_b': 5., 'sigma_b': 0.3}

model = micro21cm.BubbleModel(bubbles_ion=True, approx_small_Q=1)

# Show <dd'>, <bb'>, <bd'>
fig, axes = pl.subplots(1, 3, figsize=(12, 4))

axes[0].loglog(model.tab_R, np.abs(model.get_dd(zplot)))

P1 = [model.get_P1(RR, Q=Qplot, **kw) for RR in model.tab_R]
P2 = [model.get_P2(RR, Q=Qplot, **kw) for RR in model.tab_R]
axes[1].loglog(model.tab_R, P1)
axes[1].loglog(model.tab_R, P2, ls='--')
axes[1].loglog(model.tab_R, np.array(P1)+np.array(P2), ls='-.')
axes[1].loglog(model.tab_R, Qplot**2 * np.ones_like(model.tab_R), ls=':')

axes[0].set_title(r'$\langle d d^{\prime} \rangle$')
axes[1].set_title(r'$\langle b b^{\prime} \rangle$')
axes[2].set_title(r'$\langle b d^{\prime} \rangle$')


for ax in axes:
    ax.set_xlabel(micro21cm.labels['R'])
    ax.set_xlim(1e-2, 1e3)
    ax.set_ylim(1e-6, 3)
