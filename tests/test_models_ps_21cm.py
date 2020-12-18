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

model = micro21cm.BubbleModel(bubbles_ion=True)

# Some made-up data
data = {8.: {'k': np.array([0.1, 0.2]),
    'D_sq': np.array([1e3, 2e3]), 'err': np.array([2e2, 2e2])}}

# Make an easy plot
kw = {'k': np.logspace(-1, 0, 21), 'Ts': 3., 'Q': 0.5}
fig1, axes1 = micro21cm.plot_ps(model=model, model_kwargs=kw, data=data)

# Make a fancier multi-panel plot
kw = {'k': np.logspace(-1, 0, 21), 'Ts': np.array([2,3,4]),
    'Q': np.array([0.1, 0.2, 0.5]), 'sigma_b': np.array([0.1, 0.4]),
    'R_b': 3., 'z': 7.9}
fig2, axes2 = micro21cm.plot_ps_multi(split_by='Q', color_by='Ts', ls_by='sigma_b',
    z=7.9, model=model, model_kwargs=kw,
    data=data, fig_kwargs={'num': 2, 'figsize': (12, 4)})

#model = micro21cm.BubbleModel(bubbles_ion=True, approx_small_Q=0)
#fig2, axes2 = micro21cm.plot_ps_multi(split_by='Q', color_by='Ts', ls_by='sigma_b',
#    z=7.9, model=model, model_kwargs=kw, lw=3, axes=axes2,
#    data=hera_dr1, fig_kwargs={'num': 2, 'figsize': (12, 4)})
